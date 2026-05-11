## calculating emissions factors for maritime shipping
# Census API key is read from the CENSUS_API_KEY environment variable.
# Set it before running: export CENSUS_API_KEY="your_key_here"
# Get a free key at: https://api.census.gov/data/key_signup.html

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pycountry
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import re
from haversine import haversine, Unit
import matplotlib.cm as cm
import matplotlib.colors as colors
import requests
from tqdm import tqdm

def get_iso3(country_name):
    """
    Convert a country name string to its ISO 3166-1 alpha-3 code using pycountry.

    Required for geopandas map merges, which key on ISO3 rather than raw name strings.
    Returns None (rather than raising) for unrecognised names so callers can handle
    missing geometries gracefully.

    Parameters
    ----------
    country_name : str
        Country name as it appears in the Census trade data (e.g. 'GERMANY').

    Returns
    -------
    str or None
        Three-letter ISO3 code (e.g. 'DEU'), or None if the name is not found.
    """
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except LookupError:
        return None


def detect_header_row(filepath):
    """
    Detect whether a raw Census CSV has its column headers on row 0 or row 1.

    The Census API sometimes prepends a metadata row before the actual header,
    so this function peeks at the first line and returns the correct header row
    index for pandas to use.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to inspect.

    Returns
    -------
    int
        0 if the first line contains 'AIR_WGT_YR' (i.e. it IS the header),
        1 if the first line is a metadata row and the header is on row 1.
    """
    with open(filepath, 'r') as f:
        first_line = f.readline()
        return 0 if 'AIR_WGT_YR' in first_line else 1

def clean(hts_num, flow_type):
    """
    Clean a single raw Census trade CSV file for one HTS chapter and flow direction.

    Performs the following steps:
      - Detects the correct header row (raw downloads sometimes have a metadata row)
      - Drops aggregate/regional rows (e.g. 'EUROPEAN UNION', 'NAFTA totals')
      - Standardises country names to match pycountry's lookup table
      - Adds ISO3 country codes via get_iso3()
      - Merges port latitude/longitude from fuel_energy_info/Port_Coords.csv
      - Overwrites the original CSV with the cleaned version in place

    IMPORTANT: Only run on raw files freshly downloaded from the Census API.
    Running on already-cleaned files may corrupt the data.
    Currently hardcoded for 2024 at the HS2 level.

    Parameters
    ----------
    hts_num : str or int
        Two-digit HS chapter number (e.g. '27' or 27). Zero-padded automatically.
    flow_type : str
        'import' or 'export'.

    Returns
    -------
    pd.DataFrame or None
        Cleaned DataFrame, or None if the file is missing or malformed.
    """
    if flow_type == "import":
        filepath = f"Import_Data/imports_2024_{hts_num}.csv"
    elif flow_type == "export":
        filepath = f"Export_Data/exports_2024_{hts_num}.csv"
    else:
        raise ValueError("Invalid flow type. Choose 'import' or 'export'.")

    # Detect header row dynamically
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}: File not found.")
        return None
    
    header_row = detect_header_row(filepath)

    df = pd.read_csv(filepath, header=header_row, usecols=[1,2,3,4,5,6,7,8,10])

    if "CTY_NAME" not in df.columns:
        print(f"Skipping {hts_num} ({flow_type}): 'CTY_NAME' column not found.")
        return None


    #identifies and removes entries that are not associated with specific countries
    phrases = ["Total", "EUROPEAN UNION", "PACIFIC RIM COUNTRIES", "CAFTA-DR", "NAFTA", "TWENTY LATIN AMERICAN REPUBLICS", "OECD", "NATO", "LAFTA", "EURO AREA", "APEC", "ASEAN", "CACM",
           "NORTH AMERICA", "CENTRAL AMERICA", "SOUTH AMERICA", "EUROPE", "ASIA", "AFRICA", "OCEANIA", "MIDDLE EAST", "CARIBBEAN", "LOW VALUE", "MAIL SHIPMENTS"]
    df = df[~df.apply(lambda row: row.astype(str).str.contains("|".join(phrases), case=False, na=False).any(), axis=1)]

    #fixes some naming for countries that are not consistent with the pycountry library
    df = df.replace({
        'CHICAGO MIDWAY INT?L AIRPORT, IL': 'CHICAGO MIDWAY INTERNATIONAL AIRPORT, IL',
        'RUSSIA': 'RUSSIAN FEDERATION',
        'MACEDONIA': 'NORTH MACEDONIA',
        'MACAU': 'MACAO',
        'BURMA': 'MYANMAR',
        'REUNION': 'RÉUNION',
        'ST LUCIA': 'SAINT LUCIA',
        'ST KITTS AND NEVIS': 'SAINT KITTS AND NEVIS',
        'ST VINCENT AND THE GRENADINES': 'SAINT VINCENT AND THE GRENADINES',
        'SINT MAARTEN': 'SINT MAARTEN (DUTCH PART)',
        'CURACAO': 'CURAÇAO',
        'CONGO (KINSHASA)': 'CONGO, DEMOCRATIC REPUBLIC OF THE',
        'CONGO (BRAZZAVILLE)': 'CONGO',
        'BRUNEI': 'BRUNEI DARUSSALAM',
        "COTE D'IVOIRE": "CÔTE D'IVOIRE",
        'KOREA, SOUTH': 'KOREA, REPUBLIC OF',
        'TURKEY': 'Türkiye',
        'FALKLAND ISLANDS (ISLAS MALVINAS)': 'FALKLAND ISLANDS (MALVINAS)',
        'BRITISH INDIAN OCEAN TERRITORIES': 'BRITISH INDIAN OCEAN TERRITORY',
        'HEARD AND MCDONALD ISLANDS': 'HEARD ISLAND AND MCDONALD ISLANDS',
        'MICRONESIA': 'MICRONESIA, FEDERATED STATES OF',
        'WEST BANK ADMINISTERED BY ISRAEL': 'ISRAEL'
    })

    #applies the function to get the ISO3 code for each country
    df["CTY_ISO3"] = df["CTY_NAME"].apply(get_iso3)

    #assigns coordinates to the ports for analysis
    port_coords = pd.read_csv('fuel_energy_info/Port_Coords.csv')
    df = df.merge(port_coords, on="PORT_NAME", how="left")

    #resaves the cleaned dataframe in place of the original
    if flow_type == "import":
        df.to_csv("Import_Data/imports_2024_"+str(hts_num)+".csv", index=False)
    elif flow_type == "export":
        df.to_csv("Export_Data/exports_2024_"+str(hts_num)+".csv", index=False)

    ######also lowkey maybe need to do containerized and plain vessel... still no real documentation on that

    return df

def clean_all():
    """
    Clean all raw Census trade CSV files for all HS2 chapters (01–98, skipping 77).

    Loops through every file in Import_Data/ and Export_Data/ and calls clean()
    on each. This is the standard first step after running census_scrape.r to
    download raw trade data.

    IMPORTANT: Only run on raw files freshly downloaded from the Census API.
    Running on already-cleaned files may corrupt the data.
    Currently hardcoded for 2024 at the HS2 level.

    Returns
    -------
    None
        Files are cleaned and overwritten in place. Progress is printed to stdout.
    """
    for i in range(1, 100):
        print(str(i))
        hts_str = str(i).zfill(2)  # Ensure the HTS number is zero-padded (e.g., 01, 02, ..., 99)

        import_path = f"Import_Data/imports_2024_{hts_str}.csv"
        export_path = f"Export_Data/exports_2024_{hts_str}.csv"

        if os.path.exists(export_path):
            clean(hts_str, "export")
        else:
            print(f"Export file missing for {hts_str}, skipping.")

        if os.path.exists(import_path):
            clean(hts_str, "import")
        else:
            print(f"Import file missing for {hts_str}, skipping.")

def concat_all():
    """
    Concatenate all cleaned Import_Data and Export_Data CSVs into a single DataFrame.

    Reads every CSV from Import_Data/ and Export_Data/, tags each row with its
    flow direction ('import' or 'export') and the two-digit HTS chapter code
    extracted from the filename, then stacks everything into one combined DataFrame.

    Run this after clean_all() to produce the unified dataset used by
    general_emissions() and the analysis notebooks. The result is equivalent to
    all_flows.csv — save it with df.to_csv('all_flows.csv', index=False) if you
    want to persist it.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all trade rows plus added columns:
        - 'flow': 'import' or 'export'
        - 'hts_code': zero-padded two-digit HS chapter string (e.g. '27')
    """
    # Get all the files in the Import_Data and Export_Data directories
    import_files = [f for f in os.listdir("Import_Data") if f.endswith(".csv")]
    export_files = [f for f in os.listdir("Export_Data") if f.endswith(".csv")]

    import_dfs = []
    for file in import_files:
        df = pd.read_csv(os.path.join("Import_Data", file))
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
        df["flow"] = "import"
        # Extract HTS code from filename
        match = re.search(r"_(\d+)\.csv", file)
        if match:
            df["hts_code"] = match.group(1).zfill(2)
        import_dfs.append(df)

    export_dfs = []
    for file in export_files:
        df = pd.read_csv(os.path.join("Export_Data", file))
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
        df["flow"] = "export"
        # Extract HTS code from filename
        match = re.search(r"_(\d+)\.csv", file)
        if match:
            df["hts_code"] = match.group(1).zfill(2)
        export_dfs.append(df)

    combined_df = pd.concat(import_dfs + export_dfs, ignore_index=True)

    return combined_df


def call_files(file_name):
    """
    Load a single cleaned trade CSV and enrich it with country geometries for mapping.

    Reads the file from Import_Data/ or Export_Data/ based on the filename prefix,
    extracts the HTS chapter code, merges country geometries from the Natural Earth
    shapefile (Maps/), and computes destination centroids (dest_lon, dest_lat).
    Falls back to name-matching if ISO3 merge leaves gaps.

    Parameters
    ----------
    file_name : str
        CSV filename, e.g. 'imports_2024_27.csv' or 'exports_2024_84.csv'.
        Must start with 'imp' or 'exp'.

    Returns
    -------
    tuple: (df, hts_name, hts_num, world)
        df       : pd.DataFrame with geometry, dest_lon, dest_lat columns added
        hts_name : str, full HS chapter description (e.g. 'MINERAL FUELS...')
        hts_num  : str, zero-padded two-digit chapter code (e.g. '27')
        world    : GeoDataFrame of Natural Earth country polygons
    """
    if file_name.startswith("imp"):
        path = os.path.join( "Import_Data", file_name)
    elif file_name.startswith("exp"):
        path = os.path.join( "Export_Data", file_name)
    else:
        raise ValueError("Invalid file name. It should start with 'imp' or 'exp'.")
    
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]

    def extract_hts_num(file_name):
        hts_with_ext = file_name.split("_")[-1]  # e.g., "27.csv"
        hts_number = hts_with_ext.replace(".csv", "")  # remove the .csv
        return hts_number.zfill(2)  # Ensure it's zero-padded (e.g., "01", "02", ..., "99")

    hts_num = str(extract_hts_num(file_name))
    # maps the code number to its name
    hts_codes = {
    "01": "LIVE ANIMALS",
    "02": "MEAT AND EDIBLE MEAT OFFAL",
    "03": "FISH AND CRUSTACEANS, MOLLUSCS AND OTHER AQUATIC INVERTEBRATES",
    "04": "DAIRY PRODUCE; BIRDS' EGGS; NATURAL HONEY; EDIBLE PRODUCTS OF ANIMAL ORIGIN, NESOI",
    "05": "PRODUCTS OF ANIMAL ORIGIN, NESOI",
    "06": "LIVE TREES AND OTHER PLANTS; BULBS, ROOTS AND THE LIKE; CUT FLOWERS AND ORNAMENTAL FOLIAGE",
    "07": "EDIBLE VEGETABLES AND CERTAIN ROOTS AND TUBERS",
    "08": "EDIBLE FRUIT AND NUTS; PEEL OF CITRUS FRUIT OR MELONS",
    "09": "COFFEE, TEA, MATE AND SPICES",
    "10": "CEREALS",
    "11": "MILLING INDUSTRY PRODUCTS; MALT; STARCHES; INULIN; WHEAT GLUTEN",
    "12": "OIL SEEDS AND OLEAGINOUS FRUITS; MISCELLANEOUS GRAINS, SEEDS AND FRUITS; INDUSTRIAL OR MEDICINAL PLANTS; STRAW AND FODDER",
    "13": "LAC; GUMS; RESINS AND OTHER VEGETABLE SAPS AND EXTRACTS",
    "14": "VEGETABLE PLAITING MATERIALS AND VEGETABLE PRODUCTS, NESOI",
    "15": "ANIMAL OR VEGETABLE FATS AND OILS AND THEIR CLEAVAGE PRODUCTS; PREPARED EDIBLE FATS; ANIMAL OR VEGETABLE WAXES",
    "16": "EDIBLE PREPARATIONS OF MEAT, FISH, CRUSTACEANS, MOLLUSCS OR OTHER AQUATIC INVERTEBRATES",
    "17": "SUGARS AND SUGAR CONFECTIONERY",
    "18": "COCOA AND COCOA PREPARATIONS",
    "19": "PREPARATIONS OF CEREALS, FLOUR, STARCH OR MILK; BAKERS' WARES",
    "20": "PREPARATIONS OF VEGETABLES, FRUIT, NUTS, OR OTHER PARTS OF PLANTS",
    "21": "MISCELLANEOUS EDIBLE PREPARATIONS",
    "22": "BEVERAGES, SPIRITS AND VINEGAR",
    "23": "RESIDUES AND WASTE FROM THE FOOD INDUSTRIES; PREPARED ANIMAL FEED",
    "24": "TOBACCO AND MANUFACTURED TOBACCO SUBSTITUTES",
    "25": "SALT; SULFUR; EARTHS AND STONE; PLASTERING MATERIALS, LIME AND CEMENT",
    "26": "ORES, SLAG AND ASH",
    "27": "MINERAL FUELS, MINERAL OILS AND PRODUCTS OF THEIR DISTILLATION; BITUMINOUS SUBSTANCES; MINERAL WAXES",
    "28": "INORGANIC CHEMICALS; ORGANIC OR INORGANIC COMPOUNDS OF PRECIOUS METALS, OF RARE-EARTH METALS, OF RADIOACTIVE ELEMENTS OR OF ISOTOPES",
    "29": "ORGANIC CHEMICALS",
    "30": "PHARMACEUTICAL PRODUCTS",
    "31": "FERTILIZERS",
    "32": "TANNING OR DYEING EXTRACTS; TANNINS AND DERIVATIVES; DYES, PIGMENTS AND OTHER COLORING MATTER; PAINTS AND VARNISHES; PUTTY AND OTHER MASTICS; INKS",
    "33": "ESSENTIAL OILS AND RESINOIDS; PERFUMERY, COSMETIC OR TOILET PREPARATIONS",
    "34": "SOAP ETC.; LUBRICATING PRODUCTS; WAXES, POLISHING OR SCOURING PRODUCTS; CANDLES ETC., MODELING PASTES; DENTAL WAXES AND DENTAL PLASTER PREPARATIONS",
    "35": "ALBUMINOIDAL SUBSTANCES; MODIFIED STARCHES; GLUES; ENZYMES",
    "36": "EXPLOSIVES; PYROTECHNIC PRODUCTS; MATCHES; PYROPHORIC ALLOYS; CERTAIN COMBUSTIBLE PREPARATIONS",
    "37": "PHOTOGRAPHIC OR CINEMATOGRAPHIC GOODS",
    "38": "MISCELLANEOUS CHEMICAL PRODUCTS",
    "39": "PLASTICS AND ARTICLES THEREOF",
    "40": "RUBBER AND ARTICLES THEREOF",
    "41": "RAW HIDES AND SKINS (OTHER THAN FURSKINS) AND LEATHER",
    "42": "ARTICLES OF LEATHER; SADDLERY AND HARNESS; TRAVEL GOODS, HANDBAGS AND SIMILAR CONTAINERS; ARTICLES OF GUT (OTHER THAN SILKWORM GUT)",
    "43": "FURSKINS AND ARTIFICIAL FUR; MANUFACTURES THEREOF",
    "44": "WOOD AND ARTICLES OF WOOD; WOOD CHARCOAL",
    "45": "CORK AND ARTICLES OF CORK",
    "46": "MANUFACTURES OF STRAW, ESPARTO OR OTHER PLAITING MATERIALS; BASKETWARE AND WICKERWORK",
    "47": "PULP OF WOOD OR OTHER FIBROUS CELLULOSIC MATERIAL; RECOVERED (WASTE AND SCRAP) PAPER AND PAPERBOARD",
    "48": "PAPER AND PAPERBOARD; ARTICLES OF PAPER PULP, PAPER OR PAPERBOARD",
    "49": "PRINTED BOOKS, NEWSPAPERS, PICTURES AND OTHER PRINTED PRODUCTS; MANUSCRIPTS, TYPESCRIPTS AND PLANS",
    "50": "SILK, INCLUDING YARNS AND WOVEN FABRICS THEREOF",
    "51": "WOOL AND FINE OR COARSE ANIMAL HAIR, INCLUDING YARNS AND WOVEN FABRICS THEREOF; HORSEHAIR YARN AND WOVEN FABRIC",
    "52": "COTTON, INCLUDING YARNS AND WOVEN FABRICS THEREOF",
    "53": "VEGETABLE TEXTILE FIBERS NESOI; YARNS AND WOVEN FABRICS OF VEGETABLE TEXTILE FIBERS NESOI AND PAPER",
    "54": "MANMADE FILAMENTS, INCLUDING YARNS AND WOVEN FABRICS THEREOF",
    "55": "MANMADE STAPLE FIBERS, INCLUDING YARNS AND WOVEN FABRICS THEREOF",
    "56": "WADDING, FELT AND NONWOVENS; SPECIAL YARNS; TWINE, CORDAGE, ROPES AND CABLES AND ARTICLES THEREOF",
    "57": "CARPETS AND OTHER TEXTILE FLOOR COVERINGS",
    "58": "SPECIAL WOVEN FABRICS; TUFTED TEXTILE FABRICS; LACE; TAPESTRIES; TRIMMINGS; EMBROIDERY",
    "59": "IMPREGNATED, COATED, COVERED OR LAMINATED TEXTILE FABRICS; TEXTILE ARTICLES SUITABLE FOR INDUSTRIAL USE",
    "60": "KNITTED OR CROCHETED FABRICS",
    "61": "ARTICLES OF APPAREL AND CLOTHING ACCESSORIES, KNITTED OR CROCHETED",
    "62": "ARTICLES OF APPAREL AND CLOTHING ACCESSORIES, NOT KNITTED OR CROCHETED",
    "63": "MADE-UP TEXTILE ARTICLES NESOI; NEEDLECRAFT SETS; WORN CLOTHING AND WORN TEXTILE ARTICLES; RAGS",
    "64": "FOOTWEAR, GAITERS AND THE LIKE; PARTS OF SUCH ARTICLES",
    "65": "HEADGEAR AND PARTS THEREOF",
    "66": "UMBRELLAS, SUN UMBRELLAS, WALKING-STICKS, SEAT-STICKS, WHIPS, RIDING-CROPS AND PARTS THEREOF",
    "67": "PREPARED FEATHERS AND DOWN AND ARTICLES THEREOF; ARTIFICIAL FLOWERS; ARTICLES OF HUMAN HAIR",
    "68": "ARTICLES OF STONE, PLASTER, CEMENT, ASBESTOS, MICA OR SIMILAR MATERIALS",
    "69": "CERAMIC PRODUCTS",
    "70": "GLASS AND GLASSWARE",
    "71": "NATURAL OR CULTURED PEARLS, PRECIOUS OR SEMIPRECIOUS STONES, PRECIOUS METALS; PRECIOUS METAL CLAD METALS, ARTICLES THEREOF; IMITATION JEWELRY; COIN",
    "72": "IRON AND STEEL",
    "73": "ARTICLES OF IRON OR STEEL",
    "74": "COPPER AND ARTICLES THEREOF",
    "75": "NICKEL AND ARTICLES THEREOF",
    "76": "ALUMINUM AND ARTICLES THEREOF",
    "78": "LEAD AND ARTICLES THEREOF",
    "79": "ZINC AND ARTICLES THEREOF",
    "80": "TIN AND ARTICLES THEREOF",
    "81": "BASE METALS NESOI; CERMETS; ARTICLES THEREOF",
    "82": "TOOLS, IMPLEMENTS, CUTLERY, SPOONS AND FORKS, OF BASE METAL; PARTS THEREOF OF BASE METAL",
    "83": "MISCELLANEOUS ARTICLES OF BASE METAL",
    "84": "NUCLEAR REACTORS, BOILERS, MACHINERY AND MECHANICAL APPLIANCES; PARTS THEREOF",
    "85": "ELECTRICAL MACHINERY AND EQUIPMENT AND PARTS THEREOF; SOUND RECORDERS AND REPRODUCERS, TELEVISION RECORDERS AND REPRODUCERS, PARTS AND ACCESSORIES",
    "86": "RAILWAY OR TRAMWAY LOCOMOTIVES, ROLLING STOCK, TRACK FIXTURES AND FITTINGS, AND PARTS THEREOF; MECHANICAL ETC. TRAFFIC SIGNAL EQUIPMENT OF ALL KINDS",
    "87": "VEHICLES, OTHER THAN RAILWAY OR TRAMWAY ROLLING STOCK, AND PARTS AND ACCESSORIES THEREOF",
    "88": "AIRCRAFT, SPACECRAFT, AND PARTS THEREOF",
    "89": "SHIPS, BOATS AND FLOATING STRUCTURES",
    "90": "OPTICAL, PHOTOGRAPHIC, CINEMATOGRAPHIC, MEASURING, CHECKING, PRECISION, MEDICAL OR SURGICAL INSTRUMENTS AND APPARATUS; PARTS AND ACCESSORIES THEREOF",
    "91": "CLOCKS AND WATCHES AND PARTS THEREOF",
    "92": "MUSICAL INSTRUMENTS; PARTS AND ACCESSORIES THEREOF",
    "93": "ARMS AND AMMUNITION; PARTS AND ACCESSORIES THEREOF",
    "94": "FURNITURE; BEDDING, CUSHIONS ETC.; LAMPS AND LIGHTING FITTINGS NESOI; ILLUMINATED SIGNS, NAMEPLATES AND THE LIKE; PREFABRICATED BUILDINGS",
    "95": "TOYS, GAMES AND SPORTS EQUIPMENT; PARTS AND ACCESSORIES THEREOF",
    "96": "MISCELLANEOUS MANUFACTURED ARTICLES",
    "97": "WORKS OF ART, COLLECTORS' PIECES AND ANTIQUES",
    "98": "SPECIAL CLASSIFICATION PROVISIONS, NESOI",
    "99": "SPECIAL IMPORT REPORTING PROVISIONS, NESOI"
}

    hts_name = hts_codes[hts_num]
    path_ne = "Maps/ne_50m_admin_0_countries.shp"

    # Load world shapefile
    world = gpd.read_file(path_ne)
    world = world[["ISO_A3", "NAME", "geometry"]]  # keep ISO_A3 + NAME

    # --- First merge on ISO_A3 ---
    df = df.merge(world[["ISO_A3", "geometry"]], 
                left_on="CTY_ISO3", 
                right_on="ISO_A3", 
                how="left")

    df["CTY_NAME_CLEAN"] = df["CTY_NAME"].str.title()
    

    # --- Fallback: where geometry is null, try matching on NAME ---
    missing_mask = df["geometry"].isna()
    if missing_mask.any():
        fallback = df.loc[missing_mask].merge(
            world[["NAME", "geometry"]],
            left_on="CTY_NAME_CLEAN",  
            right_on="NAME",
            how="left",
            suffixes=("", "_world")
        )
        # fill the missing geometries
        df.loc[missing_mask, "geometry"] = fallback["geometry_world"].values

        # # Merge data with world dataset to get destination coordinates
        # world = gpd.read_file(path_ne)
        # df["CTY_ISO3"] = df["CTY_ISO3"].str.strip().str.upper() # Clean ISO3 codes
        # world = world[["ISO_A3", "geometry"]]  # Keep only ISO3 and geometry
        # df = df.merge(world, left_on="CTY_ISO3", right_on="ISO_A3", how="left")
        # # df["ISO_A3"] = df["ISO_A3"].fillna(df["CTY_ISO3"]) # Fill missing ISO_A3 with CTY_ISO3

        # print(df["geometry"].isna())

        # Extract destination centroids
        df["geometry"] = df["geometry"].apply(lambda geom: geom.centroid if geom else None)  # Ensure valid centroids
        df["dest_lon"] = df["geometry"].apply(lambda geom: geom.x if geom else None)
        df["dest_lat"] = df["geometry"].apply(lambda geom: geom.y if geom else None)

    return df, hts_name, hts_num, world

    def calc_distance(row):
        port_coords = (row["PORT_LAT"], row["PORT_LON"])
        dest_coords = (row["dest_lat"], row["dest_lon"])
        return haversine(port_coords, dest_coords, unit=Unit.KILOMETERS)

    # Apply it to each row and save in new column
    df["haversine_distance_km"] = df.apply(calc_distance, axis=1)

# mar_plot_20("exports_2024_01.csv", "Container Ship", "lsfo")
#add the load factor 
#add stowage factor
#add detour factor

#structurally having a segment for "last mile" kind of analysis




#################################
### 4 DIGIT HTS CODE STUFF ###
#################################

def fetch_trade_data(hts_2digit: str, year="2024"):
    """
    Fetch import and export trade data for all 4-digit HTS codes within an HS2 chapter.

    Calls the Census Bureau International Trade API at the HS4 level for every
    4-digit code nested under the given 2-digit chapter (e.g. chapter '39' fetches
    codes 3901–3999). Both imports and exports are fetched for each code.

    Requires the CENSUS_API_KEY environment variable to be set.

    Parameters
    ----------
    hts_2digit : str or int
        Two-digit HS chapter number (e.g. '39' or 39).
    year : str
        Reference year for the annual trade totals (default '2024').

    Returns
    -------
    dict
        Keys are '{flow_type}_{year}_{hts4}' (e.g. 'imports_2024_3901'),
        values are raw DataFrames straight from the API. Pass the result to
        build_all_master() to clean and combine into a single DataFrame.
    """
    
    hts_2digit = str(hts_2digit).zfill(2)
    results = {}
    
    census_api_key = os.environ.get("CENSUS_API_KEY")
    if not census_api_key:
        raise EnvironmentError(
            "CENSUS_API_KEY environment variable is not set. "
            "Get a free key at https://api.census.gov/data/key_signup.html "
            "and run: export CENSUS_API_KEY='your_key_here'"
        )

    base_url_template = (
        "https://api.census.gov/data/timeseries/intltrade/{trade_type}/porths"
        "?get=AIR_VAL_YR,AIR_WGT_YR,VES_VAL_YR,VES_WGT_YR,"
        "CNT_VAL_YR,CNT_WGT_YR,CTY_CODE,CTY_NAME,PORT,PORT_NAME&key={census_api_key}&"
        "{commodity_param}&YEAR&COMM_LVL=HS4&time={year}-12"
    )
    
    # Loop through 4-digit codes inside the HS2 chapter
    for code_4digit in range(int(hts_2digit) * 100, int(hts_2digit) * 100 + 100):
        code_4digit = str(code_4digit).zfill(4)
        
        # Imports and exports
        for trade_type, commodity_param in [
            ("imports", f"I_COMMODITY={code_4digit}"),
            ("exports", f"E_COMMODITY={code_4digit}")
        ]:
            
            url = base_url_template.format(
                trade_type=trade_type,
                commodity_param=commodity_param,
                year=year,
                census_api_key=census_api_key
            )
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # If the API returns an error structure, skip
                if isinstance(data, dict) and "error" in data:
                    continue
                
                # Convert JSON rows into a DataFrame
                df = pd.DataFrame(data[1:], columns=data[0])
                
                # Save in results dictionary
                key = f"{trade_type}_{year}_{code_4digit}"
                results[key] = df
            
            except Exception:
                # Skip broken URLs, timeouts, or bad responses
                continue
    
    return results

def build_all_master(data_dict):
    """
    Clean and combine a dictionary of raw HS4 trade DataFrames into one master DataFrame.

    Intended to be called directly on the output of fetch_trade_data(). Iterates over
    every key of the form '{flow_type}_{year}_{hts4}', runs cleanup_4digit() on each
    DataFrame, attaches flow type / year / HTS4 metadata columns, and concatenates
    everything into a single unified DataFrame with numeric columns cast appropriately.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by fetch_trade_data(), with keys like 'imports_2024_3901'.

    Returns
    -------
    pd.DataFrame
        Combined master DataFrame with all import and export rows plus columns:
        - 'FLOW_TYPE': 'imports' or 'exports'
        - 'YEAR': int
        - 'HTS4': int, four-digit commodity code
        All trade value/weight columns cast to numeric.
    """

    dfs = []
    
    # Regex pattern to extract flow, year, and HTS code
    pattern = re.compile(r'^(imports|exports)_(\d{4})_(\d{4})$')

    for key, df in data_dict.items():
        match = pattern.match(key)
        if not match:
            # skip non-matching or malformed keys
            continue
        
        flow_type, year, hts4 = match.groups()
        
        df_clean = cleanup_4digit(df.copy())

        # Add metadata columns
        df_clean["FLOW_TYPE"] = flow_type
        df_clean["YEAR"] = int(year)
        df_clean["HTS4"] = hts4

        dfs.append(df_clean)

    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)
    else:
        master_df = pd.DataFrame()  # fallback if empty

    #turning strings into floats
    cols = [
        'AIR_VAL_YR', 'AIR_WGT_YR',
        'VES_VAL_YR', 'VES_WGT_YR',
        'CNT_VAL_YR', 'CNT_WGT_YR',
        'YEAR', 'HTS4'
    ]

    master_df[cols] = master_df[cols].apply(pd.to_numeric, errors='coerce')

    return master_df

def cleanup_4digit(df):
    """
    Clean a single raw HS4-level trade DataFrame fetched from the Census API.

    The HS4 equivalent of clean() for use in the 4-digit analysis pipeline.
    Applies the same filtering and standardisation steps: drops aggregate/regional
    rows, normalises country names to match pycountry, adds ISO3 codes, and merges
    port coordinates. Does NOT overwrite any file — returns the cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by fetch_trade_data() for a single HS4 code.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with CTY_ISO3, PORT_LAT, and PORT_LON columns added.
    """
    #identifies and removes entries that are not associated with specific countries
    phrases = ["Total", "EUROPEAN UNION", "PACIFIC RIM COUNTRIES", "CAFTA-DR", "NAFTA", "TWENTY LATIN AMERICAN REPUBLICS", "OECD", "NATO", "LAFTA", "EURO AREA", "APEC", "ASEAN", "CACM",
           "NORTH AMERICA", "CENTRAL AMERICA", "SOUTH AMERICA", "EUROPE", "ASIA", "AFRICA", "OCEANIA", "MIDDLE EAST", "CARIBBEAN", "LOW VALUE", "MAIL SHIPMENTS"]
    df = df[~df.apply(lambda row: row.astype(str).str.contains("|".join(phrases), case=False, na=False).any(), axis=1)]

    #fixes some naming for countries that are not consistent with the pycountry library
    df = df.replace({
        'CHICAGO MIDWAY INT?L AIRPORT, IL': 'CHICAGO MIDWAY INTERNATIONAL AIRPORT, IL',
        'RUSSIA': 'RUSSIAN FEDERATION',
        'MACEDONIA': 'NORTH MACEDONIA',
        'MACAU': 'MACAO',
        'BURMA': 'MYANMAR',
        'REUNION': 'RÉUNION',
        'ST LUCIA': 'SAINT LUCIA',
        'ST KITTS AND NEVIS': 'SAINT KITTS AND NEVIS',
        'ST VINCENT AND THE GRENADINES': 'SAINT VINCENT AND THE GRENADINES',
        'SINT MAARTEN': 'SINT MAARTEN (DUTCH PART)',
        'CURACAO': 'CURAÇAO',
        'CONGO (KINSHASA)': 'CONGO, DEMOCRATIC REPUBLIC OF THE',
        'CONGO (BRAZZAVILLE)': 'CONGO',
        'BRUNEI': 'BRUNEI DARUSSALAM',
        "COTE D'IVOIRE": "CÔTE D'IVOIRE",
        'KOREA, SOUTH': 'KOREA, REPUBLIC OF',
        'TURKEY': 'Türkiye',
        'FALKLAND ISLANDS (ISLAS MALVINAS)': 'FALKLAND ISLANDS (MALVINAS)',
        'BRITISH INDIAN OCEAN TERRITORIES': 'BRITISH INDIAN OCEAN TERRITORY',
        'HEARD AND MCDONALD ISLANDS': 'HEARD ISLAND AND MCDONALD ISLANDS',
        'MICRONESIA': 'MICRONESIA, FEDERATED STATES OF',
        'WEST BANK ADMINISTERED BY ISRAEL': 'ISRAEL'
    })

    #applies the function to get the ISO3 code for each country
    df["CTY_ISO3"] = df["CTY_NAME"].apply(get_iso3)

    #assigns coordinates to the ports for analysis
    port_coords = pd.read_csv('fuel_energy_info/Port_Coords.csv')
    df = df.merge(port_coords, on="PORT_NAME", how="left")


    return df

#################################
## Shortest Feasible Distances ##
#################################

import pandas as pd
import os
from tqdm import tqdm
from haversine import haversine, Unit

def add_ocean_distances(df, cache_file='ocean_distances_cache.csv', use_cache=True):
    """
    Add straight-line (haversine) and routed ocean distances to a trade DataFrame.

    Computes haversine_distance_km for every row, then uses the `searoute` library
    to calculate realistic ocean routing distances (ship_distance_km_no_detour).
    The detour adjustment factor (ISO 14083 DAF of 1.15) is applied separately
    in general_emissions(), not here.

    Ocean routing is computationally expensive, so results are cached to a CSV by
    default. On subsequent runs, only new port-country pairs are calculated; the
    rest are loaded from cache. Set use_cache=False to force recalculation of all
    routes (useful if port coordinates have changed).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain PORT_LAT, PORT_LON, dest_lat, dest_lon columns.
    cache_file : str
        Path to the routing cache CSV (default 'ocean_distances_cache.csv').
    use_cache : bool
        Whether to use the cache (default True). Set False to recalculate all routes.

    Returns
    -------
    pd.DataFrame
        Input df with two new columns:
        - 'haversine_distance_km': great-circle distance in km
        - 'ship_distance_km_no_detour': routed ocean distance in km (pre-DAF)
    """
    def calc_haversine(row):
        port_coords = (row["PORT_LAT"], row["PORT_LON"])
        dest_coords = (row["dest_lat"], row["dest_lon"])
        return haversine(port_coords, dest_coords, unit=Unit.KILOMETERS)
    
    df["haversine_distance_km"] = df.apply(calc_haversine, axis=1)

    #shipping distances with caching
    if not use_cache:
        # Original implementation without caching
        def calculate_single_route(row):
            try:
                import searoute as sr
                origin = [row['PORT_LON'], row['PORT_LAT']]
                dest = [row['dest_lon'], row['dest_lat']]
                route = sr.searoute(origin, dest)
                
                if route and 'properties' in route:
                    return route['properties']['length']
            except:
                pass
            
            return row['haversine_distance_km'] * 1.15  # Fallback factor if routing fails
        
        tqdm.pandas(desc="Ocean routing")
        df['ship_distance_km_no_detour'] = df.progress_apply(calculate_single_route, axis=1)
        return df
    
    # WITH CACHING
    # Create route keys (round to reduce unique combinations)
    df['route_key'] = (
        df['PORT_LAT'].round(2).astype(str) + '_' + 
        df['PORT_LON'].round(2).astype(str) + '::' + 
        df['dest_lat'].round(2).astype(str) + '_' + 
        df['dest_lon'].round(2).astype(str)
    )
    
    # Load or create cache
    if os.path.exists(cache_file):
        print(f"  Loading distance cache from {cache_file}...")
        cache_df = pd.read_csv(cache_file)
        # Only treat routes with a valid (non-NaN) distance as truly cached.
        # Routes with NaN distances were written by a previous broken run and
        # must be recalculated.
        valid_cache = cache_df['ship_distance_km_no_detour'].notna()
        cached_keys = set(cache_df.loc[valid_cache, 'route_key'].values)
        n_stale = (~valid_cache).sum()
        if n_stale:
            print(f"    Found {valid_cache.sum()} valid cached routes ({n_stale} stale NaN entries will be recalculated)")
            cache_df = cache_df[valid_cache].reset_index(drop=True)
        else:
            print(f"    Found {len(cache_df)} cached routes")
    else:
        print(f"  No cache found, will create {cache_file}")
        cache_df = pd.DataFrame(columns=['route_key', 'PORT_LAT', 'PORT_LON',
                                         'dest_lat', 'dest_lon', 'ship_distance_km_no_detour'])
        cached_keys = set()

    # Identify unique routes that need calculation
    unique_routes = df[['route_key', 'PORT_LAT', 'PORT_LON',
                        'dest_lat', 'dest_lon']].drop_duplicates('route_key')
    routes_to_calc = unique_routes[~unique_routes['route_key'].isin(cached_keys)]
    
    print(f"  Total unique routes: {len(unique_routes)}")
    print(f"  Already cached: {len(unique_routes) - len(routes_to_calc)}")
    print(f"  Need to calculate: {len(routes_to_calc)}")
    
    # Calculate missing routes
    if len(routes_to_calc) > 0:
        print(f"  Calculating {len(routes_to_calc)} new ocean routes...")

        def calculate_single_route(row):
            try:
                import searoute as sr
                origin = [row['PORT_LON'], row['PORT_LAT']]
                dest = [row['dest_lon'], row['dest_lat']]
                route = sr.searoute(origin, dest)

                if route and 'properties' in route:
                    return route['properties']['length']
            except Exception as e:
                pass

            # Fallback to haversine * 1.15
            h_dist = haversine(
                (row['PORT_LAT'], row['PORT_LON']),
                (row['dest_lat'], row['dest_lon']),
                unit=Unit.KILOMETERS
            ) * 1.15
            return h_dist

        tqdm.pandas(desc="  Ocean routing")
        # Use .copy() before assigning to avoid pandas SettingWithCopyWarning —
        # routes_to_calc is a slice, so assigning directly would silently write
        # to a throwaway copy and leave the column as NaN.
        routes_to_calc = routes_to_calc.copy()
        routes_to_calc['ship_distance_km_no_detour'] = routes_to_calc.progress_apply(
            calculate_single_route, axis=1
        )
        
        # Append new results to cache
        new_cache_entries = routes_to_calc[['route_key', 'PORT_LAT', 'PORT_LON', 
                                            'dest_lat', 'dest_lon', 'ship_distance_km_no_detour']]
        cache_df = pd.concat([cache_df, new_cache_entries], ignore_index=True)
        
        # Save updated cache
        cache_df.to_csv(cache_file, index=False)
        print(f"  Cache updated: {len(cache_df)} total routes saved")
    
    # Merge cached distances back to original dataframe
    df = df.merge(
        cache_df[['route_key', 'ship_distance_km_no_detour']], 
        on='route_key', 
        how='left'
    )
    
    # Check for any missing distances
    missing_distances = df['ship_distance_km_no_detour'].isna().sum()
    if missing_distances > 0:
        print(f"  WARNING: {missing_distances} routes missing distances!")
    
    # Clean up temporary column
    df.drop('route_key', axis=1, inplace=True)
    
    return df

#############################
### Ad Valorem Tariffs ######
#############################

def process_advalorem_tariffs(hs_level=2, file_path=None, tau_col='tau', q_trim=0.05, n_bins=3):
    """
    Reads and processes ad-valorem tariff data for HS-2 or HS-4 codes.

    Parameters
    ----------
    hs_level : int
        HS code level (2 or 4). Determines default file and HS column.
    file_path : str, optional
        Full path to the .dta file. If None, defaults to:
            - 'Ad_Valorem_Data/tau_hs2_a.dta' for hs_level=2
            - 'Ad_Valorem_Data/tau_hs4_a.dta' for hs_level=4
    tau_col : str
        Name of the column containing tau estimates.
    q_trim : float
        Quantile for trimming lower/upper tails (default 0.05 for 5%).
    n_bins : int
        Number of quantile bins for tau binned medians (default 3).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - tau_truncated
        - tau_no_outliers
        - tau_binned
        - tau_median
        - error_i
        - hts_num
    """
    
    # Determine default file and HS column if not supplied
    if file_path is None:
        if hs_level == 2:
            file_path = 'Ad_Valorem_Data/tau_hs2_a.dta'
            hs_col = 'hs2'
        elif hs_level == 4:
            file_path = 'Ad_Valorem_Data/tau_hs4_a.dta'
            hs_col = 'hs4'
        else:
            raise ValueError("hs_level must be 2 or 4 if file_path not provided.")
    
    # Step 0: Read data
    df = pd.read_stata(file_path).copy()
    
    # Step 1: Truncate tau at zero
    df['tau_truncated'] = df[tau_col].clip(lower=0)
    
    # Step 2: Trim extreme outliers
    lower_q = df['tau_truncated'].quantile(q_trim)
    upper_q = df['tau_truncated'].quantile(1 - q_trim)
    df['tau_no_outliers'] = df['tau_truncated'].clip(lower=lower_q, upper=upper_q)
    
    # Step 3: Binned median
    bins = pd.qcut(df[tau_col], q=n_bins, duplicates='drop')
    df['tau_binned'] = df.groupby(bins)[tau_col].transform('median')
    
    # Step 4: Median tau
    tau_median = df[tau_col].median()
    df['tau_median'] = tau_median
    
    # Step 5: Empirical error (deviation from median)
    df['error_i'] = df['tau_no_outliers'] - tau_median
    
    # Step 6: Trim extreme errors
    lower_err = df['error_i'].quantile(q_trim)
    upper_err = df['error_i'].quantile(1 - q_trim)
    df['error_i'] = df['error_i'].clip(lower=lower_err, upper=upper_err)
    
    # Step 7: HS numeric code
    df['hts_num'] = df[hs_col].astype(int)
    
    return df

#################################
### OPTIMIZATION STUFF ##########
#################################




def general_emissions(df, fuel_type="lsfo"):
    """
    Compute maritime and aviation transport emissions for a cleaned trade DataFrame.

    This is the primary emissions model and the expected entry point for analysis.
    It should be called on the output of concat_all() (i.e. all_flows.csv) or on
    a build_all_master() result for HS4-level analysis.

    Steps performed:
      1. Parses and standardises the HTS chapter column
      2. Classifies each trade lane as Bulk Carrier, Tanker, or Container Ship
      3. Loads fuel intensity from fuel_energy_info/ based on fuel_type
      4. Merges country centroids from the Natural Earth shapefile
      5. Calculates ocean routing distances via add_ocean_distances() (with caching)
      6. Merges commodity-level stowage factors and ad-valorem tariff estimates
      7. Computes maritime emissions (gCO2eq_maritime, MtCO2eq_maritime)
         accounting for containerised vs. non-containerised weight splits
      8. Computes aviation emissions (kgCO2eq_aviation, MtCO2eq_aviation)
         using an empirically-derived log model
      9. Computes hypothetical maritime emissions for air cargo rows
         (hypo_MtCO2eq_maritime) — used as input to total_transport_cost()

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned trade DataFrame from concat_all() or build_all_master().
        Must contain Census API columns: CTY_NAME, CTY_ISO3, PORT_NAME,
        AIR_VAL_YR, AIR_WGT_YR, VES_VAL_YR, VES_WGT_YR, CNT_WGT_YR.
    fuel_type : str
        Maritime fuel scenario. One of: 'lsfo' (default), 'liquid hydrogen',
        'ammonia', 'methanol', 'FT diesel'.

    Returns
    -------
    pd.DataFrame
        Copy of input df with all intermediate and final emissions columns added.
    """

    df = df.copy()


    ############################
    # 1. HTS parsing
    ############################
    if "hts_num" not in df.columns:
        # find the HTS column
        hts_cols = [c for c in df.columns if re.match(r"(?i)^hts", c)]
        if len(hts_cols) != 1:
            raise ValueError("Exactly one HTS column required")

        hts_col = hts_cols[0]

        # Extract digits, zero-pad to at least 2 digits,
        # then take the first two digits
        df["hts_num_str"] = (
            df[hts_col]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .str.zfill(2)
            .str[:2]
        )

        # Numeric chapter code
        df["hts_num"] = df["hts_num_str"].astype("Int64")

    else:
        # Ensure we standardize even if column already exists
        df["hts_num_str"] = (
            df["hts_num"]
            .astype(str)
            .str.zfill(2)
            .str[:2]
        )

        df["hts_num"] = df["hts_num_str"].astype("Int64")

    ############################
    # 2. Vessel classification
    ############################
    bulk = {10, 11, 12, 25, 26, 30, 31, 32, 33, 34, 38, 44, 45, 72, 73, 74, 75, 76, 78, 79, 80, 81}
    tanker = {15, 27, 28, 29}

    def get_vessel_type(hts):
        if pd.isna(hts):
            return "Container Ship"
        elif int(hts) in bulk:
            return "Bulk Carrier"
        elif int(hts) in tanker:
            return "Tanker"
        else:
            return "Container Ship"
        
    df["vessel_type_pred"] = df["hts_num"].apply(get_vessel_type)

    ############################
    # 3. Fuel lookup
    ############################
    fuel_map = {
        "lsfo": 0,
        "liquid hydrogen": 1,
        "ammonia": 2,
        "methanol": 3,
        "FT diesel": 4
    }

    if fuel_type not in fuel_map:
        raise ValueError("Invalid fuel type")

    fuel_idx = fuel_map[fuel_type]
    GREET_factor = 92  # gCO2/MJ... this is WTW

    GJtnm = pd.read_csv(
        "fuel_energy_info/fuel_GJ_per_tonne_nautical_mile_lsfo_cap.csv"
    )

    intensities = {
        "Bulk Carrier":   GJtnm.iloc[fuel_idx, 4]  * 1000 / 1.852,
        "Container Ship": GJtnm.iloc[fuel_idx, 9]  * 1000 / 1.852,
        "Tanker":         GJtnm.iloc[fuel_idx, 16] * 1000 / 1.852,
    }

    ############################
    # 4. country coordinates
    ############################

    path_ne = "Maps/ne_50m_admin_0_countries.shp"

    # Load world shapefile
    world = gpd.read_file(path_ne)
    world = world[["ISO_A3", "NAME", "geometry"]]

    # --- First merge on ISO_A3 ---
    df = df.merge(world[["ISO_A3", "geometry"]], 
                left_on="CTY_ISO3", 
                right_on="ISO_A3", 
                how="left")

    df["CTY_NAME_CLEAN"] = df["CTY_NAME"].str.title()

    # --- Fallback: where geometry is null, try matching on NAME ---
    missing_mask = df["geometry"].isna()
    if missing_mask.any():
        fallback = df.loc[missing_mask].merge(
            world[["NAME", "geometry"]],
            left_on="CTY_NAME_CLEAN",  
            right_on="NAME",
            how="left",
            suffixes=("", "_world")
        )
        # fill the missing geometries
        df.loc[missing_mask, "geometry"] = fallback["geometry_world"].values

    # Extract destination centroids (MOVED OUTSIDE - applies to ALL rows)
    df["geometry"] = df["geometry"].apply(lambda geom: geom.centroid if geom else None)
    df["dest_lon"] = df["geometry"].apply(lambda geom: geom.x if geom else None)
    df["dest_lat"] = df["geometry"].apply(lambda geom: geom.y if geom else None)

    # Now calculate ocean distances with caching
    df = add_ocean_distances(df)



    ############################
    # 5. Adding stowage factors and ad-valorem tarrifs
    ############################

    ## Stowage factors ##
    #manually constructed data dictionary of stowage factor estimates
    stowage_factors = {
        1:  {"mid": 3.0},   # Live animals (AI estimate)
        2:  {"low": 2.26, "high": 3.96},   # Meat and edible meat offal From meat/seafood (manuscript)
        3:  {"low": 2.26, "high": 3.96},   # Fish and crustaceans From meat/seafood (manuscript)
        4:  {"low": 1.7,  "high": 3.96},   # Dairy, eggs, honey From otherfoodstuffs (manuscript)
        5:  {"mid": 1.5},   # Animal products n.e.s. (AI estimate)
        6:  {"mid": 2.0},   # Live trees and plants (AI estimate)
        7:  {"low": 1.7,  "high": 3.96},   # Vegetables From otherfoodstuffs (manuscript)
        8:  {"low": 1.7,  "high": 3.96},   # Fruit and nuts From otherfoodstuffs (manuscript)
        9:  {"low": 1.7,  "high": 3.96},   # Coffee, tea, spices From otherfoodstuffs (manuscript)
        10: {"low": 1.3,  "high": 1.8},   # Cereals From cereal grains (manuscript)
        11: {"low": 1.34, "high": 1.67},   # Milling products From milled grain prods. (manuscript)
        12: {"low": 1.17, "high": 3.55},   # Oil seeds From other ag prods. (manuscript)
        13: {"mid": 1.6},   # Gums, resins (AI estimate)
        14: {"mid": 2.5},   # Vegetable plaiting materials (AI estimate)
        15: {"mid": 1.1},   # Animal or vegetable fats & oils (AI estimate)
        16: {"mid": 1.5},   # Prepared meat/fish (AI estimate)
        17: {"mid": 1.17},   # Sugars From other ag prods. (manuscript)
        18: {"mid": 1.2},   # Cocoa (AI estimate)
        19: {"mid": 1.5},   # Preparations of cereals (AI estimate)
        20: {"mid": 1.4},   # Preparations of vegetables (AI estimate)
        21: {"mid": 1.4},   # Misc edible preparations (AI estimate)
        22: {"low": 1.7,  "high": 1.87},   # Beverages From alcoholic beverages (manuscript)
        23: {"low": 1.03,  "high": 2.07},   # Food industry residues From animal feed (manuscript)
        24: {"low": 3.12,  "high": 3.96},   # Tobacco From tobacco prods. (manuscript)

        25: {"low": 0.67,  "high": 1.7},   # Salt, sulfur, earths, stone From building stone, natural sands, and gravel (manuscript)
        26: {"low": 0.29,  "high": 1.15},   # Ores, slag, ash From metallic ores (manuscript)
        27: {"low": 0.6,  "mid": 0.8,  "high": 1.3},   # Mineral fuels (solid cargo) (AI + plus high end of Coal in manuscript)

        28: {"mid": 0.9},   # Inorganic chemicals (AI estimate)
        29: {"mid": 1.1},   # Organic chemicals (AI estimate)
        30: {"low": 2.0,  "high": 7.1},   # Pharmaceuticals From pharmaceuticals (manuscript)
        31: {"low": 1.0,  "high": 1.3},   # Fertilizers From fertilizers (manuscript)
        32: {"mid": 1.3},   # Tanning, dyes, paints (AI estimate)
        33: {"mid": 1.5},   # Essential oils, cosmetics (AI estimate)
        34: {"mid": 1.4},   # Soap, cleaning products (AI estimate)
        35: {"mid": 1.2},   # Albuminoidal substances (AI estimate)
        36: {"mid": 1.1},   # Explosives (AI estimate)
        37: {"mid": 1.6},   # Photographic goods (AI estimate)
        38: {"mid": 1.3},   # Misc chemical products (AI estimate)

        39: {"low": 0.67, "high": 1.15},   # Plastics From table 4 of https://www.mdpi.com/2313-4321/4/3/32
        40: {"mid": 0.56},   # Rubber From plastics/rubber (manuscript)

        41: {"low": 1.14, "high": 6.5},   # Raw hides and skins From textiles/leather (manuscript)
        42: {"low": 1.14, "high": 6.5},   # Leather articles From textiles/leather (manuscript)
        43: {"low": 1.14, "high": 6.5},   # Furskins From textiles/leather (manuscript)

        44: {"low": 0.99, "high": 3.5},   # Wood From Logs and wood prods. (manuscript)
        45: {"mid": 2.0},   # Cork (AI estimate)
        46: {"mid": 2.6},   # Straw, basketware (AI estimate)

        47: {"mid": 2.0},   # Pulp (AI estimate)
        48: {"low": 1.84, "high": 9.9},   # Paper and paperboard From paper articles (manuscript)
        49: {"low": 1.78, "high": 2.8},   # Printed matter From Newsprint/paper/printed prods (manuscript)

        50: {"mid": 2.4},   # Silk (AI estimate)
        51: {"mid": 2.2},   # Wool (AI estimate)
        52: {"mid": 2.0},   # Cotton (AI estimate)
        53: {"mid": 2.2},   # Other vegetable fibers (AI estimate)

        54: {"mid": 1.8},   # Man-made filaments (AI estimate)
        55: {"mid": 1.9},   # Man-made staple fibers (AI estimate)
        56: {"mid": 2.0},   # Wadding, felt (AI estimate)
        57: {"mid": 2.2},   # Carpets (AI estimate)
        58: {"mid": 1.9},   # Special woven fabrics (AI estimate)
        59: {"low": 1.14, "high": 6.5},   # Coated fabrics From textiles/leather (manuscript)
        60: {"low": 1.14, "high": 6.5},   # Knitted fabrics From textiles/leather (manuscript)
        61: {"low": 1.14, "high": 6.5},   # Apparel (knit) From textiles/leather (manuscript)
        62: {"low": 1.14, "high": 6.5},   # Apparel (woven) From textiles/leather (manuscript)
        63: {"low": 1.14, "high": 6.5},   # Made-up textiles From textiles/leather (manuscript)

        64: {"mid": 2.2},   # Footwear (AI estimate)
        65: {"mid": 2.6},   # Headgear (AI estimate)
        66: {"mid": 2.8},   # Umbrellas (AI estimate)
        67: {"mid": 2.6},   # Feathers, artificial flowers (AI estimate)

        68: {"mid": 1.2},   # Stone, plaster (AI estimate)
        69: {"mid": 1.1},   # Ceramics (AI estimate)
        70: {"mid": 1.98},   # Glass From nonmetal min. prods. (manuscript)

        71: {"mid": 0.6},   # Precious metals/stones (AI estimate)

        72: {"mid": 0.5},   # Iron and steel (https://www.marinesko.org/dry-bulk-chartering/stowage-factors)
        73: {"mid": 0.7},   # Articles of iron or steel (AI estimate)

        74: {"low": 0.34, "high": 1},   # Copper (https://www.marinesko.org/dry-bulk-chartering/stowage-factors)
        75: {"mid": 0.6},   # Nickel (AI estimate)
        76: {"low": 0.7, "high": 1.2},   # Aluminum (https://www.marinesko.org/dry-bulk-chartering/stowage-factors)
        78: {"low": 0.5, "high": 0.8},   # Lead (https://www.marinesko.org/dry-bulk-chartering/stowage-factors)
        79: {"low": 0.5, "high": 1.1},   # Zinc (https://www.marinesko.org/dry-bulk-chartering/stowage-factors)
        80: {"mid": 0.7},   # Tin (AI estimate)

        81: {"low": 0.34, "high": 1.04},   # Other base metals From articles-base metal (manuscript)

        82: {"mid": 1.0},   # Tools (AI estimate)
        83: {"low": 0.34, "high": 1.04},   # Misc base metal articles From articles-base metal (manuscript)

        84: {"mid": 4.25},   # Machinery From machinery (manuscript)
        85: {"low": 2.4,  "high": 2.8},   # Electrical machinery From precision instruments (manuscript)

        86: {"mid": 0.9},   # Railway stock (AI estimate)
        87: {"mid": 4.2},   # Vehicles From motorized vehicles (manuscript)
        88: {"mid": 1.0},   # Aircraft (AI estimate)
        89: {"mid": 0.8},   # Ships (AI estimate)

        90: {"low": 2.4, "high": 2.8},   # Precision instruments From precision instruments (manuscript)
        91: {"mid": 1.4},   # Clocks and watches (AI estimate)
        92: {"mid": 1.6},   # Musical instruments (AI estimate)
        93: {"mid": 1.1},   # Arms and ammunition (AI estimate)

        94: {"low": 3.1,  "high": 6.23},   # Furniture From furniture (manuscript)
        95: {"mid": 2.4},   # Toys, games (AI estimate)
        96: {"low": 1.6,  "high": 3},   # Misc manufactured articles From miscellaneous manufactured articles (manuscript)
        97: {"mid": 1.6},   # Works of art (AI estimate)
        98: {"mid": 3.0},   # Special classification provisions (AI estimate)
        99: {"mid": 2.5},   # Other articles (AI estimate)
    }
    
    #converts to df for easy merge
    stowage_factors_df = (
        pd.DataFrame.from_dict(stowage_factors, orient="index")
        .reset_index()
        .rename(columns={"index": "hts_num"})
    )
    stowage_factors_df["hts_num"] = stowage_factors_df["hts_num"].astype("Int64")
    df = df.merge(stowage_factors_df, on='hts_num', how='left')


    ## Ad valorem tariffs ##

    # first determine if hs2 or hs4 is needed
    sample_code = str(df['hts_num'].iloc[0])  # take first value
    hts_level = len(sample_code)  # 2 for HS2, 4 for HS4, etc.

    # call function to extract cleaned tariff estimates from Schaur file
    hts_processed = process_advalorem_tariffs(hs_level=hts_level)
    hts_processed = hts_processed[[
        'hts_num', 'tau_no_outliers', 'error_i'
    ]]
    df = df.merge(hts_processed, on='hts_num', how='left')

    # Compute median from the processed data
    median_tau = hts_processed['tau_no_outliers'].median()

    # Replace NaNs in the merged df
    df['tau_no_outliers'] = df['tau_no_outliers'].fillna(median_tau)
    df['error_i'] = df['error_i'].fillna(0)



 
    ############################
    # 6a. Maritime emissions
    ############################
    ship_detour = 1.15  #  ISO 14083 recommendation on DAF
    df['ship_distance_km'] = df['ship_distance_km_no_detour'] * ship_detour 

    # Convert weights
    df["VES_WGT_T"] = df["VES_WGT_YR"] / 1000
    df["CNT_WGT_T"] = df["CNT_WGT_YR"] / 1000
    df["NONCNT_WGT_T"] = (df["VES_WGT_T"] - df["CNT_WGT_T"]).clip(lower=0)

    # calculate effective weight using midpoints

    ################### 
    # TEMPORARY: testing impact of SF correction
    SF_crit = 1
    df['mid'] = df['mid'].fillna(
        df[['low', 'high']].mean(axis=1)
    )
    df['sf_multiplier'] = np.maximum(1, df['mid'] / SF_crit)
    ####################


    df["VES_WGT_eff"] = df["VES_WGT_T"] * df['sf_multiplier']
    df["CNT_WGT_eff"] = df["CNT_WGT_T"] * df['sf_multiplier']
    df["NONCNT_WGT_eff"] = (df["VES_WGT_eff"] - df["CNT_WGT_eff"]).clip(lower=0) 

    # Intensities
    container_intensity = intensities["Container Ship"]

    df["noncontainer_intensity"] = df["vessel_type_pred"].map({
        "Bulk Carrier": intensities["Bulk Carrier"],
        "Tanker": intensities["Tanker"],
        "Container Ship": intensities["Container Ship"],  # fallback
    })

    # tkm calc
    df["tkm_container"] = df["CNT_WGT_T"] * df["ship_distance_km"]
    df["tkm_noncontainer"] = df["NONCNT_WGT_T"] * df["ship_distance_km"]

    # EFFECTIVE tkm calc
    df["tkm_container_eff"] = df["CNT_WGT_eff"] * df["ship_distance_km"]
    df["tkm_noncontainer_eff"] = df["NONCNT_WGT_eff"] * df["ship_distance_km"]

    # emissions with effective mass
    df["gCO2eq_container"] = (
        GREET_factor
        * container_intensity
        * df["tkm_container_eff"]
    )

    df["gCO2eq_noncontainer"] = (
        GREET_factor
        * df["noncontainer_intensity"]
        * df["tkm_noncontainer_eff"]
    )

    df["gCO2eq_maritime"] = (
        df["gCO2eq_container"] + df["gCO2eq_noncontainer"]
    )

    df["MtCO2eq_maritime"] = df["gCO2eq_maritime"] / 1e12

    # --- Dominant vessel type label (for stowage logic later) ---
    df["vessel_type_used"] = np.where(
        df["CNT_WGT_T"] >= df["NONCNT_WGT_T"],
        "Container Ship",
        df["vessel_type_pred"]
    )
    ############################
    # 6b. Aviation emissions
    ############################
    def av_intensity(distance_km):
        d = distance_km * 1.09
        return 14.7485 - 3.3753*np.log(d) + 0.2008*(np.log(d)**2)

    df["AIR_WGT_T"] = df["AIR_WGT_YR"] / 1000

    ############
    # calculaitng the maritime volumetric mass for air cargo also (so that when we calculated what hypothetical transport cost we are more accurate)
    df["AIR_WGT_eff"] = df["AIR_WGT_T"] * df['sf_multiplier']
    df["tkm_aviation_eff"] = df["AIR_WGT_eff"] * df["haversine_distance_km"]
    ############

    df["tkm_aviation"] = df["AIR_WGT_T"] * df["haversine_distance_km"]
    #should I also multiply that by the distance adjustment factor?

    df["kgCO2_per_tkm_air"] = av_intensity(df["haversine_distance_km"])
    df["kgCO2eq_aviation"] = df["tkm_aviation"] * df["kgCO2_per_tkm_air"]

    df["MtCO2eq_aviation"] = df["kgCO2eq_aviation"] / 1e9

    ############################
    # 7. Hypothetical maritime emissions for air cargo
    # Per-row estimate of the CO2 that would be emitted if each row's
    # currently-air cargo were instead shipped by container ship along
    # the same sea route.  Used to correctly price maritime carbon costs
    # in the SCC analysis — avoids the inaccurate global-ratio proxy.
    ############################
    df["hypo_gCO2eq_maritime"] = (
        GREET_factor
        * container_intensity          # MJ / (tonne · km)
        * df["AIR_WGT_T"]              # tonnes
        * df["ship_distance_km"]       # km
    )
    df["hypo_MtCO2eq_maritime"] = df["hypo_gCO2eq_maritime"] / 1e12

    return df




################################
### Classification Things ######
################################

def draw_parameters(n):
    """
    Draw a set of randomised transport cost and speed parameters for Monte Carlo simulation.

    Samples n values for each uncertain parameter from calibrated distributions:
      - Aviation freight cost ($/tkm): Normal(0.543, 0.0543), from IATA 2025
      - Maritime freight cost ($/tkm): Uniform(0.0016, 0.0032), from Halim et al. (2018)
      - Maritime speed (km/h): Normal(29, 4) clipped to [18, 40]
      - Aviation speed (km/h): fixed at 800 (treated as deterministic)

    Called internally by monte_carlo_transport_cost(). Can also be used directly
    to inspect the parameter distributions.

    Parameters
    ----------
    n : int
        Number of draws (i.e. number of Monte Carlo simulations).

    Returns
    -------
    tuple: (av_cost, mar_cost, mar_speed, av_speed)
        av_cost   : np.ndarray of shape (n,), aviation $/tkm draws
        mar_cost  : np.ndarray of shape (n,), maritime $/tkm draws
        mar_speed : np.ndarray of shape (n,), maritime speed draws in km/h
        av_speed  : float, fixed aviation speed in km/h
    """
    # $ per tkm for aviation transport 
    # from https://www.iata.org/en/iata-repository/publications/economic-reports/global-outlook-for-air-transport-june-2025/ pg 13
    av_cost = np.random.normal(
        loc=0.543,
        scale=0.0543,
        size=n
    )

    # $ per tkm for maritime transport from Halim et al. (2018)
    mar_cost = np.random.uniform(0.0016, 0.0032, n)

    # estimate in km/h
    av_speed = 800

    # idk what distribution
    # also estimate in km/h
    mar_speed = np.clip(
        np.random.normal(loc=29, scale=4, size=n),
        18,
        40
    )

    return (
        av_cost,
        mar_cost,
        mar_speed,
        av_speed
    )



def total_transport_cost(
    df,
    av_cost,
    mar_cost,
    mar_speed,
    av_speed,
    error_draws,
    scc=0.0,
):
    """
    Compute total transport costs for both modes and classify mode-shift opportunities.

    For each trade row, calculates the total cost of the actual (aviation) shipment
    and the hypothetical cost of moving that same cargo by maritime. Cost includes
    direct freight cost ($/tkm × weight × distance) plus inventory cost (ad-valorem
    tariff × cargo value × transit days). Optionally adds a social cost of carbon
    to each mode when scc > 0.

    A row is flagged as a 'mode_shift_opportunity' if hypothetical maritime cost
    is cheaper than actual aviation cost.

    Requires df to have been processed by general_emissions() first, so that
    MtCO2eq_aviation, hypo_MtCO2eq_maritime, ship_distance_km, and tau_no_outliers
    columns are present. Called inside monte_carlo_transport_cost() for each sim.

    Parameters
    ----------
    df : pd.DataFrame
        Output of general_emissions().
    av_cost : float
        Aviation freight rate in $/tkm (one draw from draw_parameters()).
    mar_cost : float
        Maritime freight rate in $/tkm (one draw from draw_parameters()).
    mar_speed : float
        Maritime speed in km/h (one draw from draw_parameters()).
    av_speed : float
        Aviation speed in km/h (fixed draw from draw_parameters()).
    error_draws : np.ndarray
        Per-row tariff error draws sampled from the empirical error distribution.
    scc : float
        Social cost of carbon in $/tonne CO2 (default 0.0 for no carbon price).

    Returns
    -------
    pd.DataFrame
        Copy of df with cost, transit time, and classification columns added,
        including 'is_air_cheaper' and 'mode_shift_opportunity' (both 0/1 int).
    """
    df = df.copy()

    # Tariff draw = point estimate + empirical error, clipped at 0
    df['tariff_draw'] = (df['tau_no_outliers'] + error_draws).clip(lower=0)

    # Travel times
    df['aviation_time_hrs'] = df['haversine_distance_km'] / av_speed
    df['aviation_time_days'] = np.ceil(df['aviation_time_hrs'] / 24)

    df['maritime_time_hrs'] = df['ship_distance_km'] / mar_speed
    df['maritime_time_days'] = np.ceil(df['maritime_time_hrs'] / 24)

    # ── Aviation costs ────────────────────────────────────────────────
    df['direct_transport_cost_av'] = (
        (df['AIR_WGT_YR'] / 1000)
        * df['haversine_distance_km']
        * av_cost
    )
    df['incurred_transport_cost_av'] = (
        df['AIR_VAL_YR']
        * df['aviation_time_days']
        * df['tariff_draw']
    )
    df['total_transport_cost_av'] = (
        df['direct_transport_cost_av']
        + df['incurred_transport_cost_av']
    )

    # ── Hypothetical maritime costs ───────────────────────────────────
    df['hypothetical_direct_transport_cost_mar'] = (
        df['AIR_WGT_T']
        * df['ship_distance_km']
        * mar_cost
    )
    df['hypothetical_incurred_transport_cost_mar'] = (
        df['AIR_VAL_YR']
        * df['maritime_time_days']
        * df['tariff_draw']
    )
    df['hypothetical_total_transport_cost_mar'] = (
        df['hypothetical_direct_transport_cost_mar']
        + df['hypothetical_incurred_transport_cost_mar']
    )

    # ── Social Cost of Carbon add-on ──────────────────────────────────
    # When scc > 0, each mode bears its own carbon cost.
    # Aviation: uses actual aviation emissions for this row.
    # Maritime: uses hypo_MtCO2eq_maritime — the per-row hypothetical
    #   maritime emissions computed in general_emissions() — instead of
    #   the inaccurate global intensity-ratio proxy that was here before.
    #   This guarantees the SCC case can only flag MORE mode-shift
    #   opportunities than the baseline (monotonicity property).
    if scc > 0:
        df['carbon_cost_av']  = df['MtCO2eq_aviation']      * 1e6 * scc
        df['carbon_cost_mar'] = df['hypo_MtCO2eq_maritime'] * 1e6 * scc
        df['total_transport_cost_av']               += df['carbon_cost_av']
        df['hypothetical_total_transport_cost_mar'] += df['carbon_cost_mar']
    else:
        df['carbon_cost_av']  = 0.0
        df['carbon_cost_mar'] = 0.0

    # ── Classification ────────────────────────────────────────────────
    # fillna(False) handles rows where ship_distance_km is NaN (ocean routing
    # failures), which propagate NaN into costs and cause the comparison to
    # return a nullable boolean that .astype(int) cannot convert.
    df['is_air_cheaper'] = (
        df['total_transport_cost_av']
        < df['hypothetical_total_transport_cost_mar']
    ).fillna(False).astype(int)

    df['mode_shift_opportunity'] = (
        df['is_air_cheaper'] == 0
    ).fillna(False).astype(int)

    return df


def monte_carlo_transport_cost(df, n_sims=1000, scc=0.0):
    """
    Run a Monte Carlo simulation over transport cost and speed uncertainty.

    Draws n_sims sets of parameters from draw_parameters(), then calls
    total_transport_cost() once per simulation. In each simulation, per-row
    tariff uncertainty is also resampled from the empirical error distribution
    in the data. Summarises each simulation as a single row of aggregate statistics.

    Pass scc > 0 to overlay a carbon price on both modes in every simulation,
    allowing analysis of how a carbon price shifts mode-shift opportunities.
    Use scc=0 (default) for the pure-cost baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Output of general_emissions(). Must contain hts_num, error_i, and all
        columns required by total_transport_cost().
    n_sims : int
        Number of Monte Carlo draws (default 1000).
    scc : float
        Social cost of carbon in $/tonne CO2 applied to both modes (default 0.0).

    Returns
    -------
    pd.DataFrame
        One row per simulation with columns: sim, av_cost, mar_cost, mar_speed,
        scc, share_air_cheaper, mode_shift_opp_share, total_av_cost,
        total_mar_cost, error_seed.
    """
    results = []

    (av_cost, mar_cost, mar_speed, av_speed) = draw_parameters(n_sims)

    hts_error_map = (
        df[['hts_num', 'error_i']]
        .drop_duplicates()
        .set_index('hts_num')['error_i']
    )
    unique_errors = hts_error_map.values

    for i in range(n_sims):
        error_seed = i
        np.random.seed(error_seed)
        error_draws = np.random.choice(unique_errors, size=len(df), replace=True)

        out = total_transport_cost(
            df,
            av_cost=av_cost[i],
            mar_cost=mar_cost[i],
            mar_speed=mar_speed[i],
            av_speed=av_speed,
            error_draws=error_draws,
            scc=scc,
        )

        results.append({
            "sim": i,
            "av_cost": av_cost[i],
            "mar_cost": mar_cost[i],
            "mar_speed": mar_speed[i],
            "scc": scc,
            "share_air_cheaper": out['is_air_cheaper'].mean(),
            "mode_shift_opp_share": out['mode_shift_opportunity'].mean(),
            "total_av_cost": out['total_transport_cost_av'].sum(),
            "total_mar_cost": out['hypothetical_total_transport_cost_mar'].sum(),
            "error_seed": error_seed,
        })

    return pd.DataFrame(results)


def hypothetical_mode_shift_emissions(df):
    """
    Summarise emissions savings for trade rows flagged as mode-shift opportunities.

    Filters the DataFrame to rows where total_transport_cost() identified maritime
    as the cheaper option (mode_shift_opportunity == 1), then prints and returns a
    comparison of actual aviation emissions versus the hypothetical maritime emissions
    those shipments would generate if moved by sea.

    Requires df to have been processed by general_emissions() (for hypo_MtCO2eq_maritime
    and MtCO2eq_aviation) and total_transport_cost() (for mode_shift_opportunity).

    Parameters
    ----------
    df : pd.DataFrame
        Output of total_transport_cost(), which itself requires general_emissions().

    Returns
    -------
    pd.DataFrame
        Subset of df containing only mode-shift-opportunity rows, with all original
        columns retained for further analysis.
    """
    df = df.copy()

    shifts = df.loc[df['mode_shift_opportunity'] == 1].copy()

    print(f"Total hypothetical maritime emissions under mode shift: {shifts['hypo_MtCO2eq_maritime'].sum():.8f} MtCO2eq")
    print(f"Total aviation emissions for these shipments: {shifts['MtCO2eq_aviation'].sum():.4f} MtCO2eq")
    print(f"Differential emissions (aviation - hypothetical maritime): {(shifts['MtCO2eq_aviation'].sum() - shifts['hypo_MtCO2eq_maritime'].sum()):.4f} MtCO2eq")

    return shifts




