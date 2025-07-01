export OUTPUT_DIR=
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY=
export GOOGLE_API_KEY="1"

export SET_TYPE="validation"
export STRATEGY=hypertree # STRATEGY in ['direct','cot','react','reflexion','hypertree']
cd ./tools/planner
TOKENIZERS_PARALLELISM=false python sole_planning.py  \
    --set_type $SET_TYPE \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --strategy $STRATEGY \
    --reflection True