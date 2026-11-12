import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='AdaHCL: Adaptive Hypergraph Contrastive Learning for Online Course Recommendation')
	# Optimization
	parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
	parser.add_argument('--lr_decay', default=0.98, type=float, help='learning rate decay per epoch')
	parser.add_argument('--lr_min', default=5e-4, type=float, help='minimum learning rate')
	parser.add_argument('--batch', default=4096, type=int, help='training batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--reg', default=1e-5, type=float, help='L2 regularization weight (gamma)')
	parser.add_argument('--epoch', default=20, type=int, help='number of epochs')
	# Model architecture
	parser.add_argument('--latdim', default=128, type=int, help='embedding dimension')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of GCN/HGCN layers')
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
	parser.add_argument('--topk', default=20, type=int, help='K of top-K evaluation')
	# Data
	parser.add_argument('--data', default='student', type=str, help='dataset name: student | mooc')
	# Loss weights
	parser.add_argument('--ssl_reg', default=0.1, type=float, help='weight for pairwise contrastive loss')
	parser.add_argument('--ib_reg', type=float, default=1e-2, help='weight for information bottleneck')
	parser.add_argument('--triple_reg', type=float, default=1e-6, help='weight for triple contrastive loss (alpha)')
	parser.add_argument('--margin', type=float, default=1.0, help='margin in triple loss')
	# Contrastive learning
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning (tau)')
	parser.add_argument('--tstEpoch', default=1, type=int, help='test every N epochs')
	# Denoising / L0
	parser.add_argument('--lambda0', type=float, default=1e-4, help='weight for L0 loss (beta)')
	parser.add_argument('--gamma', type=float, default=-0.45, help='hard concrete lower bound')
	parser.add_argument('--zeta', type=float, default=1.05, help='hard concrete upper bound')
	parser.add_argument('--init_temperature', type=float, default=2.0, help='initial temperature for annealing')
	parser.add_argument('--temperature_decay', type=float, default=0.98, help='temperature decay rate')
	parser.add_argument('--eps', type=float, default=1e-3, help='epsilon for Adam optimizer')
	# Environment
	parser.add_argument('--gpu', default=-1, type=int, help='GPU id, -1 for auto')
	parser.add_argument('--seed', type=int, default=2026, help='random seed')

	return parser.parse_args()
args = ParseArgs()
