import torch as t
import torch.nn.functional as F


def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	"""InfoNCE contrastive loss between two sets of embeddings."""
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -t.log(nume / deno)

def tripleLoss(anchor, positive, negative, margin=1.0):
	"""Triple contrastive loss for AdaHCL.

	L_triple = max(0, ||z_anchor - z_positive||^2 - ||z_anchor - z_negative||^2 + margin)

	Args:
		anchor:   Embeddings from the adaptive graph view (Z_graph).
		positive: Embeddings from the adaptive hypergraph view (Z_hyper).
		negative: Embeddings from the denoised graph view (Z_denoise) for different nodes.
		margin:   Minimum distance gap between positive and negative samples.
	"""
	pos_dist = t.sum((anchor - positive) ** 2, dim=-1)
	neg_dist = t.sum((anchor - negative) ** 2, dim=-1)
	loss = t.clamp(pos_dist - neg_dist + margin, min=0.0)
	return loss