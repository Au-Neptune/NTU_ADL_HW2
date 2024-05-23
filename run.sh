mkdir "tmp"
echo "[step log] make tmp folder"
python ./inference.py --model ./model --max_length 64 --beam_size 4 --top_k 20 --top_p 0.85 --test_data $1 --output_dir ./tmp/
echo "[step log] inference done"
mv ./tmp/submission.jsonl $2
echo "[step log] ALL DONE"