# Colab A100 data setup (run from /content/code). KAGGLE_API_TOKEN (asuudane) comes from Colab Secrets via the notebook cell.
pip -q install -U timm kaggle
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2
IN=/content/in; OUT=/content/out; mkdir -p $IN $OUT /content/w
kaggle datasets download asuudane/datascape-small-512 -p $IN/small512 --unzip -q
kaggle datasets download asuudane/datascape-aux -p $IN/aux --unzip -q
(cd $IN/small512 && for z in *.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm "$z"; done)
kaggle datasets download asuudane/datascape-colab-out -p $OUT --unzip -q || echo "no previous colab-out (first run)"
find $IN -name Aman__Aman_00001.jpg | grep -q . || { echo "DATA MISSING"; exit 1; }
echo "setup ok: $(find $IN -name '*.jpg' | wc -l) images, previous outputs: $(ls $OUT | wc -l)"
