export SET_TYPE=validation
export EVALUATION_FILE_PATH="/home/rqgui/TravelPlanner/outputs/first_test.jsonl"
cd evaluation
CUDA_VISIBLE_DEVICES=3  python eval.py --set_type $SET_TYPE --evaluation_file_path $EVALUATION_FILE_PATH