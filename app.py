# from flask import Flask, render_template, request
# import pickle
# from pathlib import Path
# from datetime import datetime

# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# MODEL_PATH = Path(__file__).with_name("yield_prediction_model.pkl")
# model = None
# model_load_error = None
# model_info = {}
# try:
#     with MODEL_PATH.open("rb") as f:
#         model = pickle.load(f)
#     st = MODEL_PATH.stat()
#     model_info = {
#         "path": str(MODEL_PATH),
#         "size_bytes": st.st_size,
#         "modified": datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds"),
#         "type": type(model).__name__,
#     }
# except Exception as e:
#     model_load_error = f"{type(e).__name__}: {e}"


# def _infer_feature_names(m):
#     if m is None:
#         return []

#     # sklearn estimators often expose training feature names
#     names = getattr(m, "feature_names_in_", None)
#     if names is not None:
#         return [str(x) for x in list(names)]

#     # fall back to number of features, if available
#     n = getattr(m, "n_features_in_", None)
#     if isinstance(n, (int, np.integer)) and int(n) > 0:
#         return [f"x{i}" for i in range(1, int(n) + 1)]

#     return ["x1"]


# FEATURES = _infer_feature_names(model)

# # Adjust these based on how your model was trained
# CATEGORICAL_FEATURES = {
#     "State_Name",
#     "District_Name",
#     "Season",
#     "Crop",
# }


# def _extract_category_options(m, feature_names):
#     """
#     If the pickled model is a sklearn Pipeline that contains a OneHotEncoder,
#     extract the known categories so the UI can offer valid choices.
#     """
#     try:
#         preprocess = getattr(m, "named_steps", {}).get("preprocess")
#         if preprocess is None:
#             return {}

#         cat = None
#         try:
#             cat = preprocess.named_transformers_.get("cat")
#         except Exception:
#             cat = None

#         categories = getattr(cat, "categories_", None)
#         if categories is None:
#             return {}

#         # Try to determine which input columns the encoder corresponds to
#         cat_features = None
#         for name, transformer, cols in getattr(preprocess, "transformers_", []):
#             if name == "cat":
#                 cat_features = list(cols) if cols is not None else None
#                 break
#         if not cat_features:
#             # Fall back to intersection with our known categorical set
#             cat_features = [f for f in feature_names if f in CATEGORICAL_FEATURES]

#         options = {}
#         for f, cats in zip(cat_features, categories, strict=False):
#             if f in feature_names:
#                 options[f] = [str(x) for x in list(cats)]
#         return options
#     except Exception:
#         return {}


# CATEGORY_OPTIONS = _extract_category_options(model, FEATURES)


# def _sanity_predictions(m, options):
#     """
#     Quick check to confirm predictions change with inputs.
#     Uses the first known category for each categorical feature and varies Area.
#     """
#     try:
#         if m is None or not options:
#             return {}
#         required = ["State_Name", "District_Name", "Season", "Crop"]
#         if not all(options.get(k) for k in required):
#             return {}

#         row = {
#             "State_Name": options["State_Name"][0],
#             "District_Name": options["District_Name"][0],
#             "Season": options["Season"][0],
#             "Crop": options["Crop"][0],
#         }
#         X1 = pd.DataFrame([{**row, "Area": 10.0}])
#         X2 = pd.DataFrame([{**row, "Area": 200.0}])
#         p1 = float(np.asarray(m.predict(X1)).ravel()[0])
#         p2 = float(np.asarray(m.predict(X2)).ravel()[0])
#         return {"example_row": row, "pred_area_10": p1, "pred_area_200": p2}
#     except Exception as e:
#         return {"error": f"{type(e).__name__}: {e}"}


# SANITY = _sanity_predictions(model, CATEGORY_OPTIONS)


# @app.route("/")
# def home():
#     if model_load_error:
#         return (
#             "Flask is working, but the model failed to load.<br><br>"
#             f"Expected model at: {MODEL_PATH}<br>"
#             f"Error: {model_load_error}"
#         ), 500
#     return render_template(
#         "index.html",
#         features=FEATURES,
#         prediction=None,
#         error=None,
#         options=CATEGORY_OPTIONS,
#         values={},
#         model_info=model_info,
#         sanity=SANITY,
#     )


# @app.route("/predict", methods=["POST"])
# def predict():
#     if model_load_error:
#         return (
#             "Model failed to load.<br><br>"
#             f"Expected model at: {MODEL_PATH}<br>"
#             f"Error: {model_load_error}"
#         ), 500

#     row = {}
#     values = {}
#     for name in FEATURES:
#         raw = request.form.get(name, "")
#         values[name] = raw

#         # Basic required check
#         if raw == "":
#             return render_template(
#                 "index.html",
#                 features=FEATURES,
#                 prediction=None,
#                 error=f"Missing value for '{name}'",
#             ), 400

#         if name in CATEGORICAL_FEATURES:
#             # Keep categorical features as strings for encoders
#             allowed = CATEGORY_OPTIONS.get(name)
#             if allowed is not None and raw not in allowed:
#                 return render_template(
#                     "index.html",
#                     features=FEATURES,
#                     prediction=None,
#                     error=(
#                         f"'{name}' value {raw!r} is not in the model's known categories. "
#                         "Please select a value from the suggestions list."
#                     ),
#                     options=CATEGORY_OPTIONS,
#                     values=values,
#                 ), 400
#             row[name] = raw
#         else:
#             # Numeric features are parsed as floats
#             try:
#                 row[name] = float(raw)
#             except ValueError:
#                 return render_template(
#                     "index.html",
#                     features=FEATURES,
#                     prediction=None,
#                     error=f"Invalid number for '{name}': {raw!r}",
#                 ), 400

#     # Use a DataFrame so sklearn pipelines with column transformers can
#     # handle mixed numeric/categorical inputs based on column names.
#     X = pd.DataFrame([row])
#     try:
#         y = model.predict(X)
#         pred = float(np.asarray(y).ravel()[0])
#     except Exception as e:
#         return render_template(
#             "index.html",
#             features=FEATURES,
#             prediction=None,
#             error=f"Prediction failed: {type(e).__name__}: {e}",
#             options=CATEGORY_OPTIONS,
#             values=values,
#         ), 500

#     return render_template(
#         "index.html",
#         features=FEATURES,
#         prediction=pred,
#         error=None,
#         options=CATEGORY_OPTIONS,
#         values=values,
#         model_info=model_info,
#         sanity=SANITY,
#     )

# if __name__ == "__main__":
#     app.run(debug=True)



















"""
SmartAgri AI — Yield Prediction Flask App
==========================================
Fixes applied:
  1. Removed strict "value not in allowed" validation — values from the
     dropdown are always valid; strict checking was rejecting legitimate inputs.
  2. Single, correct postprocess_prediction(raw_pred, area) — no duplicate.
  3. State→District map built from model encoder data with Indian district hints.
  4. Clean _render_kwargs helper eliminates repeated keyword arguments.
"""

from flask import Flask, render_template, request, jsonify
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

app = Flask(__name__)

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).with_name("yield_prediction_model.pkl")
model = None
model_load_error = None
model_info = {}

try:
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    st = MODEL_PATH.stat()
    model_info = {
        "size_kb": round(st.st_size / 1024, 1),
        "modified": datetime.fromtimestamp(st.st_mtime).strftime("%d %b %Y"),
        "type": type(model).__name__,
    }
except Exception as e:
    model_load_error = f"{type(e).__name__}: {e}"


# ── Feature names ───────────────────────────────────────────────────────────
def _infer_feature_names(m):
    if m is None:
        return []
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]
    n = getattr(m, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and int(n) > 0:
        return [f"x{i}" for i in range(1, int(n) + 1)]
    return ["x1"]


FEATURES = _infer_feature_names(model)
CATEGORICAL_FEATURES = {"State_Name", "District_Name", "Season", "Crop"}


# ── Pull dropdown options from the sklearn pipeline's OneHotEncoder ─────────
def _extract_category_options(m, feature_names):
    try:
        pre = getattr(m, "named_steps", {}).get("preprocess")
        if pre is None:
            return {}
        cat_tf = None
        try:
            cat_tf = pre.named_transformers_.get("cat")
        except Exception:
            pass
        cats = getattr(cat_tf, "categories_", None)
        if cats is None:
            return {}
        # Identify which columns correspond to the cat transformer
        cat_cols = None
        for nm, _, cols in getattr(pre, "transformers_", []):
            if nm == "cat":
                cat_cols = list(cols)
                break
        if not cat_cols:
            cat_cols = [f for f in feature_names if f in CATEGORICAL_FEATURES]
        options = {}
        for col, col_cats in zip(cat_cols, cats):
            if col in feature_names:
                options[col] = [str(v) for v in col_cats]
        return options
    except Exception:
        return {}


CATEGORY_OPTIONS = _extract_category_options(model, FEATURES)


# ── Build State → [District] mapping ────────────────────────────────────────
def _build_state_district_map(options):
    """
    Match the model's district list against a hardcoded Indian state→district
    lookup using normalised substring matching.
    """
    states    = options.get("State_Name", [])
    districts = options.get("District_Name", [])
    if not states or not districts:
        return {}

    def norm(s):
        return s.strip().upper().replace(" ", "").replace("_", "").replace("-", "")

    # Comprehensive state → district hints (covers the standard Kaggle
    # Indian Crop Production dataset used for most such models)
    HINTS = {
        "ANDHRAPRADESH": [
            "ADILABAD","ANANTAPUR","CHITTOOR","EASTGODAVARI","GUNTUR","KARIMNAGAR",
            "KHAMMAM","KRISHNA","KURNOOL","MAHABUBNAGAR","MEDAK","NALGONDA","NELLORE",
            "NIZAMABAD","PRAKASAM","RANGAREDDI","SRIKAKULAM","VISAKHAPATNAM",
            "VIZIANAGARAM","WARANGAL","WESTGODAVARI","CUDDAPAH","HYDERABAD",
        ],
        "ARUNACHALPRADESH": [
            "CHANGLANG","DIBANGVALLEY","EASTKAMENG","EASTSIANG","KURUNGKUMEY",
            "LOHIT","LOWERSUBANSIRI","PAPUMPARE","TAWANG","TIRAP","UPPERSUBANSIRI",
            "UPPERSIANG","WESTKAMENG","WESTSIANG",
        ],
        "ASSAM": [
            "BARPETA","BONGAIGAON","CACHAR","DARRANG","DHEMAJI","DHUBRI","DIBRUGARH",
            "DIMAHASAO","GOALPARA","GOLAGHAT","HAILAKANDI","JORHAT","KAMRUP",
            "KARBIANGLONG","KARIMGANJ","KOKRAJHAR","LAKHIMPUR","MARIGAON",
            "NAGAON","NALBARI","SIVASAGAR","SONITPUR","TINSUKHIA",
        ],
        "BIHAR": [
            "ARARIA","ARWAL","AURANGABAD","BANKA","BEGUSARAI","BHAGALPUR","BHOJPUR",
            "BUXAR","DARBHANGA","EASTCHAMPARAN","GAYA","GOPALGANJ","JAMUI","JEHANABAD",
            "KAIMUR","KATIHAR","KHAGARIA","KISHANGANJ","LAKHISARAI","MADHEPURA",
            "MADHUBANI","MUNGER","MUZAFFARPUR","NALANDA","NAWADA","PATNA","PURNIA",
            "ROHTAS","SAHARSA","SAMASTIPUR","SARAN","SHEIKHPURA","SHEOHAR",
            "SITAMARHI","SIWAN","SUPAUL","VAISHALI","WESTCHAMPARAN",
        ],
        "CHATTISGARH": [
            "BASTAR","BIJAPUR","BILASPUR","DANTEWADA","DHAMTARI","DURG","JANJGIR",
            "JASHPUR","KANKER","KAWARDHA","KORBA","KOREA","MAHASAMUND","NARAYANPUR",
            "RAIGARH","RAIPUR","RAJNANDGAON","SURGUJA",
        ],
        "GOA": ["NORTHGOA","SOUTHGOA"],
        "GUJARAT": [
            "AHMEDABAD","AMRELI","ANAND","BANASKANTHA","BHARUCH","BHAVNAGAR","DAHOD",
            "DANGS","GANDHINAGAR","JAMNAGAR","JUNAGADH","KHEDA","KUTCH","MAHESANA",
            "NARMADA","NAVSARI","PANCHMAHAL","PATAN","PORBANDAR","RAJKOT",
            "SABARKANTHA","SURAT","SURENDRANAGAR","TAPI","VADODARA","VALSAD",
        ],
        "HARYANA": [
            "AMBALA","BHIWANI","FARIDABAD","FATEHABAD","GURGAON","HISAR","JHAJJAR",
            "JIND","KAITHAL","KARNAL","KURUKSHETRA","MAHENDRAGARH","MEWAT","PALWAL",
            "PANCHKULA","PANIPAT","REWARI","ROHTAK","SIRSA","SONIPAT","YAMUNANAGAR",
        ],
        "HIMACHALPRADESH": [
            "BILASPUR","CHAMBA","HAMIRPUR","KANGRA","KINNAUR","KULLU","LAHUL",
            "MANDI","SHIMLA","SIRMAUR","SOLAN","UNA",
        ],
        "JAMMUANDKASHMIR": [
            "ANANTNAG","BADGAM","BANDIPORA","BARAMULA","DODA","GANDERBAL","JAMMU",
            "KARGIL","KATHUA","KISHTWAR","KUPWARA","LEH","POONCH","PULWAMA",
            "RAJOURI","RAMBAN","REASI","SAMBA","SHOPIAN","SRINAGAR","UDHAMPUR",
        ],
        "JHARKHAND": [
            "BOKARO","CHATRA","DEOGHAR","DHANBAD","DUMKA","EASTSINGHBHUM","GARHWA",
            "GIRIDIH","GODDA","GUMLA","HAZARIBAG","JAMTARA","KHUNTI","KODERMA",
            "LATEHAR","LOHARDAGA","PAKUR","PALAMU","RAMGARH","RANCHI","SAHIBGANJ",
            "SERAIKELA","SIMDEGA","WESTSINGHBHUM",
        ],
        "KARNATAKA": [
            "BAGALKOT","BANGALORE","BANGARAPETE","BELGAUM","BELLARY","BIDAR","BIJAPUR",
            "CHAMARAJANAGAR","CHIKKABALLAPUR","CHIKMAGALUR","CHITRADURGA",
            "DAKSHINAKANNADA","DAVANAGERE","DHARWAD","GADAG","GULBARGA","HASSAN",
            "HAVERI","KODAGU","KOLAR","KOPPAL","MANDYA","MYSORE","RAICHUR",
            "RAMANAGARA","SHIMOGA","TUMKUR","UDUPI","UTTARAKANNADA","YADGIR",
        ],
        "KERALA": [
            "ALAPPUZHA","ERNAKULAM","IDUKKI","KANNUR","KASARAGOD","KOLLAM",
            "KOTTAYAM","KOZHIKODE","MALAPPURAM","PALAKKAD","PATHANAMTHITTA",
            "THIRUVANANTHAPURAM","THRISSUR","WAYANAD",
        ],
        "MADHYAPRADESH": [
            "ALIRAJPUR","ANUPPUR","ASHOKNAGAR","BALAGHAT","BARWANI","BETUL","BHIND",
            "BHOPAL","BURHANPUR","CHHATARPUR","CHHINDWARA","DAMOH","DATIA","DEWAS",
            "DHAR","DINDORI","GUNA","GWALIOR","HARDA","HOSHANGABAD","INDORE",
            "JABALPUR","JHABUA","KATNI","KHANDWA","KHARGONE","MANDLA","MANDSAUR",
            "MORENA","NARSINGHPUR","NEEMUCH","PANNA","RAISEN","RAJGARH","RATLAM",
            "REWA","SAGAR","SATNA","SEHORE","SEONI","SHAHDOL","SHAJAPUR","SHEOPUR",
            "SHIVPURI","SIDHI","SINGRAULI","TIKAMGARH","UJJAIN","UMARIA","VIDISHA",
        ],
        "MAHARASHTRA": [
            "AHMEDNAGAR","AKOLA","AMRAVATI","AURANGABAD","BEED","BHANDARA","BULDHANA",
            "CHANDRAPUR","DHULE","GADCHIROLI","GONDIA","HINGOLI","JALGAON","JALNA",
            "KOLHAPUR","LATUR","MUMBAI","NAGPUR","NANDED","NANDURBAR","NASHIK",
            "OSMANABAD","PARBHANI","PUNE","RAIGAD","RATNAGIRI","SANGLI","SATARA",
            "SINDHUDURG","SOLAPUR","THANE","WARDHA","WASHIM","YAVATMAL",
        ],
        "MANIPUR": [
            "BISHNUPUR","CHANDEL","CHURACHANDPUR","IMPHALEAST","IMPHALWEST",
            "SENAPATI","TAMENGLONG","THOUBAL","UKHRUL",
        ],
        "MEGHALAYA": [
            "EASTGAROHILLS","EASTKHASIHILLS","JAINTHIAHILLS","RIBHOI",
            "SOUTHGAROHILLS","WESTGAROHILLS","WESTKHASIHILLS",
        ],
        "MIZORAM": ["AIZAWL","CHAMPHAI","KOLASIB","LAWNGTLAI","LUNGLEI","MAMIT","SAIHA","SERCHHIP"],
        "NAGALAND": ["DIMAPUR","KIPHIRE","KOHIMA","LONGLENG","MOKOKCHUNG","MON","PEREN","PHEK","TUENSANG","WOKHA","ZUNHEBOTO"],
        "ODISHA": [
            "ANGUL","BALANGIR","BALASORE","BARGARH","BHADRAK","BOUDH","CUTTACK",
            "DEOGARH","DHENKANAL","GAJAPATI","GANJAM","JAGATSINGHPUR","JAJPUR",
            "JHARSUGUDA","KALAHANDI","KANDHAMAL","KENDRAPARA","KENDUJHAR","KHORDHA",
            "KORAPUT","MALKANGIRI","MAYURBHANJ","NABARANGAPUR","NAYAGARH","NUAPADA",
            "PURI","RAYAGADA","SAMBALPUR","SONAPUR","SUNDARGARH",
        ],
        "PUDUCHERRY": ["KARAIKAL","MAHE","PUDUCHERRY","YANAM"],
        "PUNJAB": [
            "AMRITSAR","BARNALA","BATHINDA","FARIDKOT","FATEHGARHSAHIB","FAZILKA",
            "FEROZEPUR","GURDASPUR","HOSHIARPUR","JALANDHAR","KAPURTHALA","LUDHIANA",
            "MANSA","MOGA","MUKTSAR","NAWANSHAHR","PATHANKOT","PATIALA","ROOPNAGAR",
            "SANGRUR","SASNAGAR","TARNTARAN",
        ],
        "RAJASTHAN": [
            "AJMER","ALWAR","BANSWARA","BARAN","BARMER","BHARATPUR","BHILWARA",
            "BIKANER","BUNDI","CHITTORGARH","CHURU","DAUSA","DHOLPUR","DUNGARPUR",
            "GANGANAGAR","HANUMANGARH","JAIPUR","JAISALMER","JALORE","JHALAWAR",
            "JHUNJHUNU","JODHPUR","KARAULI","KOTA","NAGAUR","PALI","PRATAPGARH",
            "RAJSAMAND","SAWAIMADHOPUR","SIKAR","SIROHI","TONK","UDAIPUR",
        ],
        "SIKKIM": ["EASTDIST","NORTHDIST","SOUTHDIST","WESTDIST"],
        "TAMILNADU": [
            "ARIYALUR","CHENNAI","COIMBATORE","CUDDALORE","DHARMAPURI","DINDIGUL",
            "ERODE","KANCHIPURAM","KANNIYAKUMARI","KARUR","KRISHNAGIRI","MADURAI",
            "NAGAPATTINAM","NAMAKKAL","NILGIRIS","PERAMBALUR","PUDUKKOTTAI",
            "RAMANATHAPURAM","SALEM","SIVAGANGA","THANJAVUR","THENI","THOOTHUKUDI",
            "TIRUCHIRAPPALLI","TIRUNELVELI","TIRUPPUR","TIRUVALLUR","TIRUVANNAMALAI",
            "TIRUVARUR","VELLORE","VILLUPURAM","VIRUDHUNAGAR",
        ],
        "TELANGANA": [
            "ADILABAD","HYDERABAD","KARIMNAGAR","KHAMMAM","MAHABUBNAGAR","MEDAK",
            "NALGONDA","NIZAMABAD","RANGAREDDI","WARANGAL",
        ],
        "TRIPURA": ["DHALAI","NORTHTRIPURA","SOUTHTRIPURA","WESTTRIPURA"],
        "UTTARPRADESH": [
            "AGRA","ALIGARH","ALLAHABAD","AMBEDKARNAGAR","AMETHI","AMROHA",
            "AURAIYA","AZAMGARH","BAGHPAT","BAHRAICH","BALLIA","BALRAMPUR",
            "BANDA","BARABANKI","BAREILLY","BASTI","BIJNOR","BUDAUN","BULANDSHAHR",
            "CHANDAULI","CHITRAKOOT","DEORIA","ETAH","ETAWAH","FAIZABAD",
            "FARRUKHABAD","FATEHPUR","FIROZABAD","GAUTAMBUDDHANAGAR","GHAZIABAD",
            "GHAZIPUR","GONDA","GORAKHPUR","HAMIRPUR","HAPUR","HARDOI","HATHRAS",
            "JALAUN","JAUNPUR","JHANSI","KANNAUJ","KANPURDEHAT","KANPURNAGAR",
            "KASGANJ","KAUSHAMBI","KUSHINAGAR","LAKHIMPURKHERI","LALITPUR",
            "LUCKNOW","MAHARAJGANJ","MAHOBA","MAINPURI","MATHURA","MAU","MEERUT",
            "MIRZAPUR","MORADABAD","MUZAFFARNAGAR","PILIBHIT","PRATAPGARH",
            "RAEBARELI","RAMPUR","SAHARANPUR","SAMBHAL","SANTKABIR NAGAR",
            "SHAHJAHANPUR","SHAMLI","SHRAVASTI","SIDDHARTHNAGAR","SITAPUR",
            "SONBHADRA","SULTANPUR","UNNAO","VARANASI",
        ],
        "UTTARAKHAND": [
            "ALMORA","BAGESHWAR","CHAMOLI","CHAMPAWAT","DEHRADUN","HARIDWAR",
            "NAINITAL","PAURIGARHWAL","PITHORAGARH","RUDRAPRAYAG","TEHRIGARHWAL",
            "UDHAMSINGHNAGAR","UTTARKASHI",
        ],
        "WESTBENGAL": [
            "BANKURA","BARDHAMAN","BIRBHUM","COOCHBEHAR","DAKSHINDINAJPUR",
            "DARJEELING","HOOGHLY","HOWRAH","JALPAIGURI","KOLKATA","MALDAH",
            "MEDINIPUR","MURSHIDABAD","NADIA","NORTH24PARGANAS","PURULIA",
            "SOUTH24PARGANAS","UTTARDINAJPUR",
        ],
    }

    norm_to_orig = {norm(d): d for d in districts}
    result = {}

    for state in states:
        sn = norm(state)
        hint_list = []
        for hstate, hlist in HINTS.items():
            # Match if normed strings share a significant substring
            if hstate in sn or sn in hstate or sn[:6] in hstate:
                hint_list = hlist
                break

        matched = []
        for hd in hint_list:
            hdn = norm(hd)
            for dn, dorig in norm_to_orig.items():
                if hdn in dn or dn in hdn:
                    if dorig not in matched:
                        matched.append(dorig)

        result[state] = sorted(matched) if matched else sorted(districts)

    return result


STATE_DISTRICT_MAP = _build_state_district_map(CATEGORY_OPTIONS)


# ── Diagnose model output scale ─────────────────────────────────────────────
def _diagnose_model(m, options):
    """
    Detects:
      use_exp  — model outputs log(yield); apply exp() to get real value
      is_rate  — model outputs yield/ha;   multiply by area for total tonnes
    """
    diag = {"use_exp": False, "is_rate": True, "error": None}
    try:
        if m is None or not options:
            return diag
        required = ["State_Name", "District_Name", "Season", "Crop"]
        if not all(options.get(k) for k in required):
            return diag

        base = {
            "State_Name":    options["State_Name"][0],
            "District_Name": options["District_Name"][0],
            "Season":        options["Season"][0],
            "Crop":          options["Crop"][0],
        }

        areas = [10.0, 1_000.0, 100_000.0]
        raws  = []
        for a in areas:
            X = pd.DataFrame([{**base, "Area": a}])
            raws.append(float(np.asarray(m.predict(X)).ravel()[0]))

        spread = max(raws) - min(raws)

        # Narrow predictions in plausible log range → log-space model
        if spread < 5 and -2 <= min(raws) <= 20:
            diag["use_exp"] = True
            working = [float(np.exp(r)) for r in raws]
        else:
            working = list(raws)

        # area grew 10000×; if prediction grew < 50× it's a rate model
        ratio = working[-1] / working[0] if working[0] != 0 else 1.0
        diag["is_rate"] = ratio < 50

    except Exception as e:
        diag["error"] = str(e)
    return diag


DIAGNOSIS   = _diagnose_model(model, CATEGORY_OPTIONS)
USE_EXP     = DIAGNOSIS["use_exp"]
IS_RATE     = DIAGNOSIS["is_rate"]


# ── Post-process raw prediction → human-readable result ─────────────────────
def postprocess_prediction(raw_pred: float, area: float) -> dict:
    """
    Always takes exactly TWO arguments: raw model output + area in hectares.
    Returns yield_per_ha (t/ha) and total_production (tonnes).
    """
    value = float(np.exp(raw_pred)) if USE_EXP else float(raw_pred)

    if IS_RATE:
        yield_per_ha     = value
        total_production = value * area
    else:
        total_production = value
        yield_per_ha     = (value / area) if area > 0 else 0.0

    return {
        "yield_per_ha":     round(yield_per_ha, 4),
        "total_production": round(total_production, 2),
        "area":             area,
    }


# ── Shared template context builder ─────────────────────────────────────────
def _ctx(**extra):
    base = dict(
        features        = FEATURES,
        result          = None,
        error           = None,
        options         = CATEGORY_OPTIONS,
        values          = {},
        model_info      = model_info,
        use_exp         = USE_EXP,
        is_rate         = IS_RATE,
        state_district_map = STATE_DISTRICT_MAP,
    )
    base.update(extra)
    return base


# ── API: districts for a given state ────────────────────────────────────────
@app.route("/api/districts")
def api_districts():
    state = request.args.get("state", "").strip()
    if state:
        districts = STATE_DISTRICT_MAP.get(state)
        if districts:
            return jsonify(districts)
    return jsonify(CATEGORY_OPTIONS.get("District_Name", []))


# ── Home ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    if model_load_error:
        return (
            f"<h2>Model load error</h2><pre>{model_load_error}</pre>"
            f"<p>Expected: {MODEL_PATH}</p>"
        ), 500
    return render_template("index.html", **_ctx())


# ── Predict ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error:
        return f"<h2>Model load error</h2><pre>{model_load_error}</pre>", 500

    row    = {}
    values = {}

    for name in FEATURES:
        raw = request.form.get(name, "").strip()
        values[name] = raw

        # ── Missing field check ──────────────────────────────────────────
        if raw == "":
            return render_template(
                "index.html",
                **_ctx(error=f"Please fill in the '{name.replace('_',' ')}' field.", values=values)
            ), 400

        # ── Type coercion ────────────────────────────────────────────────
        if name in CATEGORICAL_FEATURES:
            # Accept whatever the dropdown sent — do NOT reject unknown strings.
            # The model/encoder will raise a proper error if it truly can't handle it.
            row[name] = raw
        else:
            try:
                row[name] = float(raw)
            except ValueError:
                return render_template(
                    "index.html",
                    **_ctx(error=f"'{name.replace('_',' ')}' must be a number. Got: {raw!r}", values=values)
                ), 400

    # ── Run model prediction ─────────────────────────────────────────────
    X = pd.DataFrame([row])
    try:
        raw_pred = float(np.asarray(model.predict(X)).ravel()[0])
        area_val = float(row.get("Area", 1.0))
        result   = postprocess_prediction(raw_pred, area_val)
    except Exception as e:
        return render_template(
            "index.html",
            **_ctx(error=f"Prediction error: {type(e).__name__}: {e}", values=values)
        ), 500

    return render_template("index.html", **_ctx(result=result, values=values))


if __name__ == "__main__":
    app.run(debug=True)