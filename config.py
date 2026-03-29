import os
import torch
 
# ── Paths (EC2) ───────────────────────────────────────────────────────────────
PATH          = '~/manga_ocr'
BASE_DIR      = os.path.expanduser(PATH)
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED= os.path.join(BASE_DIR, "data", "processed")
 
DATASET_ROOT  = os.path.join(DATA_PROCESSED, "detection")
RECOG_JSON    = os.path.join(DATA_PROCESSED, "recognition", "labels.json")
 
MODELS_DIR        = os.path.join(BASE_DIR, "models")
DETECT_MODEL_DIR  = os.path.join(MODELS_DIR, "detect")
RECOG_MODEL_DIR   = os.path.join(MODELS_DIR, "recog")
LOG_DIR           = os.path.join(BASE_DIR, "logs")
 
for d in [MODELS_DIR, DETECT_MODEL_DIR, RECOG_MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
 
# ── Image sizes ───────────────────────────────────────────────────────────────
IMG_SIZE  = 640
IMG_HEIGHT  = 640
IMG_WIDTH  = 640
RECOG_HEIGHT   = 32
RECOG_WIDTH   = 128
 
# ── Character set — paste output of charset_builder.py here ──────────────────
CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"
    "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
    "きゃきゅきょしゃしゅしょちゃちゅちょにゃにゅにょひゃひゅひょみゃみゅみょりゃりゅりょぎゃぎゅぎょじゃじゅじょびゃびゅびょぴゃぴゅぴょ"
    "キャキュキョシャシュショチャチュチョコニャニュニョヒャヒュヒョミャミュミョリャリュリョギャギュギョジャジュジョビャビュビョピャピュピョ"
    "ウィウェウォヴァヴィヴヴェヴォファフィフェフォチェシェジェティディトゥドゥ"
    "日事一人見本子出年大言学分中記会新月時行気報思上語自者生文明情国朝用書私手間小合方社検目前入索関作特何女今体動集発最内投下知地場別話部化告法広来田理物開全説聞表連無対的な高校感心以成名業長家稿定実山近現後金覧男画性度数立彼問二意能個僕通面回代利経使車編同平音読少食道世結力楽真品考公野込所不当取在電愛外載向美返仕版際変示親治政島権解他先掲川三口機風東市付持式加界要信多更活選題屋論郎済有身線味著顔売空続第様海始校英勝母次点正科京術転録初葉相約終育住白等声俺字決登北案天産切都格主県資井十元戦想原指円店死容流過保町足介料安着健調芸違研古参番館受歩値歴笑引形村然果和究重西議奈直確強注組良辞好試可水必送万"
    "。、「」『』・！（）？〜：；ー"
    # Add your Japanese characters here after running charset_builder.py
)
NUM_CLASSES = len(CHARSET) + 1
 
# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS_DETECT  = 50    # No time limit on EC2 — use full epochs
EPOCHS_RECOG   = 100
LR             = 1e-3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
 
# ── Thresholds ────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.4
IOU_THRESHOLD  = 0.5
 
# ── Checkpoint resume ─────────────────────────────────────────────────────────
RESUME_DETECT  = False
RESUME_RECOG   = False
 
print(f"Config loaded. Device: {DEVICE}  |  Charset size: {NUM_CLASSES - 1}")
