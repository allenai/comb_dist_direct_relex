# run allennlp training on beaker

dataset="ds_vqxds2hmemyl:/work/data/freebase"  # freebase
# dataset="ds_zkhfwh47wxdo:/work/data/umls"  # umls
config_file="allennlp_config/freebase.json"

for SEED in 13570 13670 13470 #  13970 13070 13170 13270 13370 14070 14170 14270 14370 14470 14570 14670 14770 14870 14970 15070 15170 # list more than one seed to run more than one run
do
    for elmo in false
    do
        for cnn_size in 100
        do
            for dropout in 0.1
            do
                for distmult in true
                do
                    for attention in false
                    do
                        for binary_loss_weight in 0
                        do
                            for sent_loss in 0 # 2 #  0 0.5 1 2 4 8 16 32 64 # 21 # 2 4 8 16 32 64 128 #   0.5 1 2 4 8 16 32 64 21  # 0.5 1 7 14 21
                            do
                                for supervised in none # all #  none all # selected positive none
                                do
                                    for attention_weights in none # none sigmoid #  none softmax sigmoid norm_sigmoid
                                    do
                                        for attention_agg in max # avg max max_add
                                        do
                                            for train_on_supervised in false
                                            do
                                                for random_pos_embedding_init in false
                                                do
                                                    for pcnn in false
                                                    do
                                                        for reduce_lambda_every in 0 #  9634 # 0
                                                        do
                                                            for negative in 100 # 2 50
                                                            do
                                                                for bag_size in 25 # 25 50
                                                                do


# SEED=13470
# attention_weights=norm_sigmoid
# ff_before_alpha=false
# attention_agg=max
# sent_loss=0
# supervised=none


PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`

export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export negative=$negative
export bag_size=$bag_size
export ff_maxpool=false
export with_bag_size=false
export weighted_maxpool=false
export elmo=$elmo

if [[ "$elmo" == true ]]; then
    export batch_size=4
else
    export batch_size=32
fi

if [[ "$sent_loss" == 0 ]]; then
    export ff_before_alpha=false
else
    export ff_before_alpha=true
fi

# if [[ "$sent_loss" == 0 ]]; then
#     export supervised=none
#     export weighted_maxpool=false
# fi

export cnn_size=$cnn_size
export dropout=$dropout
export distmult=$distmult
export attention=$attention
export binary_loss_weight=$binary_loss_weight
export sent_loss=$sent_loss
export supervised=$supervised
export weighted_maxpool=$weighted_maxpool
export train_on_supervised=$train_on_supervised
export ff_maxpool=$ff_maxpool
export with_bag_size=$with_bag_size
export reduce_lambda_every=$reduce_lambda_every
export random_pos_embedding_init=$random_pos_embedding_init
export pcnn=$pcnn
export attention_weights=$attention_weights
export ff_before_alpha=$ff_before_alpha
export attention_agg=$attention_agg


if [ "$sent_loss" != 0 ] && [ "$supervised" == none ]; then
    echo "Wrong args 1. Ignore <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,"
    continue
fi

if [ "$sent_loss" == 0 ] && [ "$ff_before_alpha" == true ]; then
    echo "Wrong args 2. Ignore <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,"
    continue
fi


# if [ "$attention_weights" == none ] && [ "$ff_before_alpha" == true ]; then
#     echo "Wrong args 3. Ignore <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,"
#     continue
# fi


# TODO: remove this for new experiments
# if [ "$sent_loss" == 0 ] && [ "$supervised" != none ]; then
#     echo "Ignore. Don't run this for now ............."
#     continue
# fi

# echo "elmo=$elmo", "batch_size=$batch_size", "cnn_size=$cnn_size", "dropout=$dropout", "distmult=$distmult", \
#      "sent_loss=$sent_loss", "supervised=$supervised", "weighted_maxpool=$weighted_maxpool", \
#      "train_on_supervised=$train_on_supervised", "random_pos_embedding_init=$random_pos_embedding_init", "pcnn=$pcnn", \
#      "reduce_lambda_every=$reduce_lambda_every", "negative=$negative", "bag_size=$bag_size", \
#      "attention_weights=$attention_weights", "ff_before_alpha=$ff_before_alpha", "attention_agg=$attention_agg"

echo "sent_loss=$sent_loss", "supervised=$supervised", "attention_weights=$attention_weights", "ff_before_alpha=$ff_before_alpha", "attention_agg=$attention_agg"

# continue  # delete this continue for the experiment to be submitted to beaker
# remember to change the desc below
python scripts/run_with_beaker.py $config_file --source $dataset --desc 'naacl2019-freebase-avg' \
    --env "negative=$negative" --env "bag_size=$bag_size" --env "elmo=$elmo" --env "batch_size=$batch_size" \
    --env "cnn_size=$cnn_size" --env "dropout=$dropout" --env "distmult=$distmult" --env "attention=$attention" --env "binary_loss_weight=$binary_loss_weight" \
    --env "training_set=$training_set" --env "sent_loss=$sent_loss" --env "supervised=$supervised" --env  "weighted_maxpool=$weighted_maxpool" \
    --env "train_on_supervised=$train_on_supervised" --env "ff_maxpool=$ff_maxpool" --env "with_bag_size=$with_bag_size" \
    --env "SEED=$SEED" --env "PYTORCH_SEED=$PYTORCH_SEED" --env "NUMPY_SEED=$NUMPY_SEED" --env "reduce_lambda_every=$reduce_lambda_every" \
    --env "random_pos_embedding_init=$random_pos_embedding_init" --env "pcnn=$pcnn" \
    --env "attention_weights=$attention_weights" --env "ff_before_alpha=$ff_before_alpha" --env "attention_agg=$attention_agg"

                                                                done
                                                            done
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
