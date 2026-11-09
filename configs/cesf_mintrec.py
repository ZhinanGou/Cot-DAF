class Param():

    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """Hyper-parameters for the final 6-granularity dual-anchor fusion on
        MIntRec (the original single-label intent dataset)."""
        hyper_parameters = {
            # common parameter
            'eval_monitor': ['acc'],
            'train_batch_size': [16],
            'eval_batch_size': [8],
            'test_batch_size': [8],
            'wait_patience': [5],
            'num_train_epochs': [40],
            # train parameters
            'warmup_proportion': [0.01],
            'lr': [9e-6],
            'weight_decay': [0.05],
            # model parameters
            'weight_dropout': [0.2],        # capacity-control dropout in the classifier
            'cross_num_heads': [16],
            'dst_feature_dims': [1024],
            'cross_dp_rate': [0.2],
            'n_levels_cross': [1],
            'self_num_heads': [8],
            'n_levels_self': [1],
        }

        return hyper_parameters
