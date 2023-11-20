from argparse import Namespace

#######################################################################
# TODO: Design your own neural network
# Set hyperparameters here
#######################################################################
HP = Namespace(
    batch_size=32,
    lr=5e-4,
    momentum=0.9,
    lr_decay=0.99,
    optim_type="adam",
    l2_reg=0.0,
    epochs=8,
    do_batchnorm=True,
    p_dropout=0.2
)
#######################################################################
