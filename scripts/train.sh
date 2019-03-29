# Run allennlp training locally
config_file="allennlp_config/config.json"

SEED=13270
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED


# change the following two variables to make the problem smaller for debugging
export negative_exampels_percentage=100 # Values < 100 will randomely drop some of the negative examples
export max_bag_size=25  # keep only the top `max_bag_size` sentences in each bag and drop the rest


# reader configurations
export with_direct_supervision=true  # false for distant supervision only


# model configurations
export cnn_size=100
export dropout_weight=0.1  # dropout weight after word embeddings
export with_entity_embeddings=true  # false for no entity embeddings

export sent_loss_weight=1  # 0, 0.5, 1, 2, 4, 8, 16, 32, 64
export attention_weight_fn=sigmoid  # uniform, softmax, sigmoid, norm_sigmoid
export attention_aggregation_fn=max  # avg, max


# trainer configurations
export batch_size=128


allennlp train $config_file --include-package relex -s $1 $2
