# run allennlp training on beaker

dataset="ds_vqxds2hmemyl:/work/data/freebase"  # freebase
config_file="allennlp_config/config.json"

for SEED in 13570 # 13670 13470
do
    for sent_loss_weight in 0 # 2 #  0 0.5 1 2 4 8 16 32 64 # 21 # 2 4 8 16 32 64 128 #   0.5 1 2 4 8 16 32 64 21  # 0.5 1 7 14 21
    do
        for with_direct_supervision in false # false true false # none # all #  none all # selected positive none
        do
            for attention_weight_fn in uniform # softmax sigmoid
            do
                for attention_aggregation_fn in max # avg
                do

PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED
export negative_exampels_percentage=100  # Values < 100 will randomely drop some of the negative examples
export max_bag_size=25  # keep only the top `max_bag_size` sentences in each bag and drop the rest
export cnn_size=100
export dropout_weight=0.1  # dropout weight after word embeddings
export with_entity_embeddings=true  # false for no entity embeddings
export batch_size=256


if [[ "$with_direct_supervision" == true ]]; then
    if [[ "$sent_loss_weight" == 0 ]]; then
        echo Ignore: "sent_loss_weight=$sent_loss_weight", "with_direct_supervision=$with_direct_supervision"
        continue
    fi
fi

if [[ "$with_direct_supervision" == false ]]; then
    if [[ "$sent_loss_weight" != 0 ]]; then
        echo Ignore: "sent_loss_weight=$sent_loss_weight", "with_direct_supervision=$with_direct_supervision"
        continue
    fi
fi


export sent_loss_weight=$sent_loss_weight
export with_direct_supervision=$with_direct_supervision
export attention_weight_fn=$attention_weight_fn
export attention_aggregation_fn=$attention_aggregation_fn


echo "sent_loss_weight=$sent_loss_weight", "with_direct_supervision=$with_direct_supervision", "attention_weight_fn=$attention_weight_fn", "attention_aggregation_fn=$attention_aggregation_fn"

# continue  # delete this continue for the experiment to be submitted to beaker
# remember to change the desc below
python scripts/run_with_beaker.py $config_file --source $dataset --desc 'naacl2019-freebase-avg' \
    --env "SEED=$SEED" --env "PYTORCH_SEED=$PYTORCH_SEED" --env "NUMPY_SEED=$NUMPY_SEED" \
    --env "negative_exampels_percentage=$negative_exampels_percentage" --env "max_bag_size=$max_bag_size" --env "cnn_size=$cnn_size" \
    --env "dropout_weight=$dropout_weight" --env "with_entity_embeddings=$with_entity_embeddings" --env "batch_size=$batch_size" \
    --env "sent_loss_weight=$sent_loss_weight" --env "with_direct_supervision=$with_direct_supervision" \
    --env "attention_weight_fn=$attention_weight_fn" --env "attention_aggregation_fn=$attention_aggregation_fn"
                done
            done
        done
    done
done
