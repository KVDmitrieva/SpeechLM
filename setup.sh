echo "Install requirements"
pip install -r requirements.txt

echo "Download SpeechLM checkpoint"
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/1azAQ_0AiwYSbmpwyTECCcraYKMkHBCxi/view?usp=sharing"