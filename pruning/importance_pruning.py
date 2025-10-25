"""


References
==========

(1) Gradient Sensitivity
===================

(|w · ∂L/∂w| or |∂L/∂w|) — used in early pruning and saliency methods.

LeCun, Yann, John S. Denker, and Sara A. Solla.
“Optimal Brain Damage.” Advances in Neural Information Processing Systems 2 (1990): 598–605.
https://proceedings.neurips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html

(They first proposed pruning based on gradient and curvature sensitivity — ∂L/∂w importance.)

(2) Fisher Information (Empirical Fisher ≈ E[g²])
=========================================

Kirkpatrick, James, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwińska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell.
“Overcoming Catastrophic Forgetting in Neural Networks.” Proceedings of the National Academy of Sciences 114, no. 13 (2017): 3521–3526.
https://doi.org/10.1073/pnas.1611835114

(Introduced Elastic Weight Consolidation, where Fisher information estimates parameter importance — foundational use of empirical Fisher ≈ E[g²].)

(3) Hessian Diagonal (Hutchinson Estimator, Optimal Brain Damage/Surgeon)
=================================================================

LeCun, Yann, John S. Denker, and Sara A. Solla.
“Optimal Brain Damage.” Advances in Neural Information Processing Systems 2 (1990): 598–605.

Hassibi, Babak, and David G. Stork.
“Second Order Derivatives for Network Pruning: Optimal Brain Surgeon.” Advances in Neural Information Processing Systems 5 (1993): 164–171.

(OBD and OBS introduce Hessian-based pruning — using diag(H) or full H⁻¹ to estimate sensitivity of removing each parameter.)

(4) Layer-Wise Relevance Propagation (LRP, ε-Rule)
============================================

Bach, Sebastian, Alexander Binder, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller, and Wojciech Samek.
“On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation.” PLOS ONE 10, no. 7 (2015): e0130140.
https://doi.org/10.1371/journal.pone.0130140

(Introduces LRP with ε-rule for relevance redistribution in linear and convolutional layers.)

(5) Data-Driven Saliency Metrics
===========================

(a) Mean(|activation|) per neuron/filter

Hu, Hengyuan, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang.
“Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures.” arXiv preprint arXiv:1607.03250 (2016).
https://arxiv.org/abs/1607.03250

(Uses average activation magnitude as pruning criterion.)

(b) Mean(|activation × grad_activation|) — activation gradient saliency

Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman.
“Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps.” arXiv preprint arXiv:1312.6034 (2013).
https://arxiv.org/abs/1312.6034

(Introduced gradient-based saliency maps — foundation for activation-gradient relevance.)

(c) Grad-CAM–style saliency (activation × gradient for conv feature maps)

Selvaraju, Ramprasaath R., Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.
“Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.” Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2017): 618–626.
https://doi.org/10.1109/ICCV.2017.74
"""


import torch
import torch.nn as nn
from collections import defaultdict
import math
import random

# -----------------------------
# Utilities
# -----------------------------
def flatten_tensor(t: torch.Tensor):
    return t.detach().view(-1)

def named_weight_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and "bias" not in n]

# -----------------------------
# Core estimator class
# -----------------------------
class ImportanceEstimator:
    def __init__(self, model: nn.Module, device=None, dtype=torch.float32):
        self.model = model
        self.device = device or (next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu'))
        self.model.to(self.device)
        self.dtype = dtype

    # -------------------------
    # 1) gradient sensitivity
    # -------------------------
    def gradient_sensitivity(self, dataloader, criterion, num_batches=1, return_per_param=False):
        """
        Compute average absolute gradient and average abs(weight * grad) per parameter.
        Returns dict: {param_name: {'abs_grad': tensor, 'w_absgrad': tensor}}
        """
        self.model.train()  # ensure BN/Dropout behave like training
        accum_abs_grad = defaultdict(float)
        accum_w_absgrad = defaultdict(float)
        counts = 0

        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                abs_grad = param.grad.detach().abs()
                w_absgrad = (param.detach().abs() * abs_grad)
                accum_abs_grad[name] = accum_abs_grad.get(name, 0) + abs_grad.clone().to('cpu')
                accum_w_absgrad[name] = accum_w_absgrad.get(name, 0) + w_absgrad.clone().to('cpu')
            counts += 1

        # average
        out = {}
        for name in accum_abs_grad:
            avg_abs_grad = accum_abs_grad[name] / counts
            avg_w_absgrad = accum_w_absgrad[name] / counts
            out[name] = {'abs_grad': avg_abs_grad, 'w_absgrad': avg_w_absgrad}
            if not return_per_param:
                # reduce to per-filter (out_channels) for Conv2d or per-neuron for Linear
                param = dict(self.model.named_parameters())[name]
                if param.ndim == 4:  # conv: (out, in, kH, kW) -> per-filter: reduce dims 1,2,3
                    out[name]['abs_grad_filter'] = avg_abs_grad.view(param.shape[0], -1).sum(dim=1)
                    out[name]['w_absgrad_filter'] = avg_w_absgrad.view(param.shape[0], -1).sum(dim=1)
                elif param.ndim == 2:  # linear: (out, in)
                    out[name]['abs_grad_neuron'] = avg_abs_grad.view(param.shape[0], -1).sum(dim=1)
                    out[name]['w_absgrad_neuron'] = avg_w_absgrad.view(param.shape[0], -1).sum(dim=1)
        return out

    # -------------------------
    # 2) empirical Fisher
    # -------------------------
    def empirical_fisher(self, dataloader, criterion, num_batches=1):
        """
        Estimate Fisher information per parameter: E[ g^2 ] where g = grad log p(y|x;theta)
        For classification with CE, use grads of loss.
        """
        self.model.train()
        accum_g2 = {}
        counts = 0
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is None: 
                    continue
                g2 = (param.grad.detach() ** 2).to('cpu')
                accum_g2[name] = accum_g2.get(name, 0) + g2
            counts += 1

        for name in list(accum_g2.keys()):
            accum_g2[name] = accum_g2[name] / counts  # average
        # Provide per-filter/neuron sums too for convenience
        out = {}
        for name, g2 in accum_g2.items():
            param = dict(self.model.named_parameters())[name]
            if param.ndim == 4:
                out[name] = {'fisher_param': g2, 'fisher_filter': g2.view(param.shape[0], -1).sum(dim=1)}
            elif param.ndim == 2:
                out[name] = {'fisher_param': g2, 'fisher_neuron': g2.view(param.shape[0], -1).sum(dim=1)}
            else:
                out[name] = {'fisher_param': g2}
        return out

    # -------------------------
    # 3) Hessian diagonal (Hutchinson)
    # -------------------------
    def hessian_diag_hutchinson(self, dataloader, criterion, num_batches=1, num_samples=10):
        """
        Estimate diagonal of Hessian of loss w.r.t parameters using Hutchinson estimator.
        diag(H) ≈ E_v [ v * H v ] with v Rademacher (+-1).
        Returns per-parameter diag estimate.
        """
        self.model.train()
        param_list = [p for n, p in self.model.named_parameters() if p.requires_grad and 'bias' not in n]
        names = [n for n, p in self.model.named_parameters() if p.requires_grad and 'bias' not in n]

        # initialize accumulators on CPU to avoid OOM on GPU for large models
        accum_diag = {n: torch.zeros_like(p.detach().cpu()) for n, p in self.model.named_parameters() if p.requires_grad and 'bias' not in n}
        total_samples = 0

        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches: 
                break
            x, y = x.to(self.device), y.to(self.device)

            for s in range(num_samples):
                # create Rademacher vectors v matching parameter shapes
                v = []
                for n, p in self.model.named_parameters():
                    if not p.requires_grad or 'bias' in n:
                        v.append(None)
                    else:
                        rv = torch.randint(0, 2, p.shape, device=self.device, dtype=p.dtype) * 2 - 1  # +-1
                        v.append(rv)

                # forward + backward to compute grads
                self.model.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                grads = torch.autograd.grad(loss, [p for n, p in self.model.named_parameters() if p.requires_grad and 'bias' not in n], create_graph=True)

                # compute Hv: directional second derivative: grad(grads dot v) wrt params
                # First compute inner product grads · v
                grads_dot_v = 0
                g_idx = 0
                for n, p in self.model.named_parameters():
                    if not p.requires_grad or 'bias' in n:
                        continue
                    grads_dot_v = grads_dot_v + (grads[g_idx] * v[g_idx]).sum()
                    g_idx += 1

                # now compute grad of grads_dot_v wrt params -> this yields H v
                Hv_list = torch.autograd.grad(grads_dot_v, [p for n, p in self.model.named_parameters() if p.requires_grad and 'bias' not in n], retain_graph=False)

                # accumulate v * Hv (on CPU)
                idx = 0
                for n, p in self.model.named_parameters():
                    if not p.requires_grad or 'bias' in n:
                        continue
                    est = (v[idx].detach() * Hv_list[idx].detach()).to('cpu')  # element-wise
                    accum_diag[n] += est
                    idx += 1
                total_samples += 1

        # average over samples and batches
        for n in accum_diag:
            accum_diag[n] = accum_diag[n] / total_samples

        # OBD score 0.5 * H_ii * w_i^2
        obd = {}
        for n in accum_diag:
            w = dict(self.model.named_parameters())[n].detach().to('cpu')
            obd[n] = 0.5 * accum_diag[n] * (w ** 2)
        return {'hessian_diag': accum_diag, 'obd_score': obd}

    # -------------------------
    # 4) Layer-wise Relevance Propagation (LRP) simple ε-rule
    # -------------------------
    def lrp_epsilon(self, dataloader, num_batches=1, eps=1e-6):
        """
        Simple LRP epsilon-rule for Linear and Conv2d layers:
            R_j = sum_i ( (a_j * w_ji) / (sum_j a_j * w_ji + eps) ) * R_i
        We'll compute relevance from output logits (use target logits as initial relevance)
        Returns per-layer relevance maps (summed per neuron/filter).
        NOTE: This is a simplified implementation for demonstration.
        """
        # Save activations and weights via forward hooks
        activations = {}
        handles = []

        def get_hook(name):
            def hook(module, inp, out):
                activations[name] = out.detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handles.append(module.register_forward_hook(get_hook(name)))

        self.model.eval()
        relevance_per_layer = defaultdict(float)
        counts = 0
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)  # forward to populate activations
            probs = torch.softmax(out, dim=1)
            # start relevance at output logits (or class score)
            # shape = (batch, classes) -> choose target class logit as relevance
            batch_size = x.size(0)
            R = torch.zeros_like(out)  # relevance at output logits
            for b in range(batch_size):
                R[b, y[b]] = out[b, y[b]].detach()  # use class logit as initial relevance

            # Backpropagate relevance layer by layer (reverse order of saved activations)
            # Convert activations keys into modules order
            layer_names = list(activations.keys())
            # We'll process in reverse order
            R_current = R
            for lname in reversed(layer_names):
                act = activations[lname]  # activation of this layer output
                module = dict(self.model.named_modules())[lname]
                W = None
                if isinstance(module, nn.Linear):
                    W = module.weight.detach()  # (out, in)
                    # compute contribution: a @ W^T  -> out
                    z = (act.unsqueeze(-1) * W.t().unsqueeze(0)).sum(dim=1)  # approximate broadcast
                    # simplified redistribution:
                    denom = z.sum(dim=1, keepdim=True) + eps
                    message = (z / denom) * R_current.sum(dim=1, keepdim=True)
                    # summarize per-neuron relevance: sum over batch
                    relevance_per_layer[lname] = relevance_per_layer.get(lname, 0) + message.abs().sum().cpu()
                elif isinstance(module, nn.Conv2d):
                    # For conv: approximate by channel-wise contributions
                    # act: (B, C_out, H, W). We'll compute channel-wise relevance by proportionality to channel activations
                    a = act.detach()
                    denom = a.sum(dim=(1,2,3), keepdim=True) + eps
                    channel_rel = (a / denom) * R_current.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                    # sum absolute relevance per channel
                    ch_sum = channel_rel.abs().sum(dim=(0,2,3)).cpu()
                    # accumulate
                    relevance_per_layer[lname] = relevance_per_layer.get(lname, 0) + ch_sum
                # For next step, set R_current to something smaller (we keep it simple)
                R_current = R_current  # simplified: not iteratively precise

            counts += 1

        for k in relevance_per_layer:
            relevance_per_layer[k] = relevance_per_layer[k] / max(1, counts)

        # remove hooks
        for h in handles:
            h.remove()

        return dict(relevance_per_layer)

    # -------------------------
    # 5) Data-driven saliency metrics (activations & activation-grad)
    # -------------------------
    def activation_based_metrics(self, dataloader, num_batches=1):
        """
        Records activations for Linear and Conv layers and computes:
          - mean_abs_activation per filter/neuron
          - mean_abs_activation_times_grad (requires backward)
        For activation_grad we run a backward on a scalar loss (sum of logits for the batch)
        """
        act_sums = defaultdict(float)
        act_abs_sums = defaultdict(float)
        act_count = defaultdict(int)

        # Hook to collect activations and gradients w.r.t activations
        acts = {}
        handles = []
        grads_of_acts = {}

        def forward_hook(name):
            def hook(module, inp, out):
                acts[name] = out
            return hook

        def backward_hook(name):
            def hook(module, grad_in, grad_out):
                # grad_out is a tuple; take first
                grads_of_acts[name] = grad_out[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(module.register_forward_hook(forward_hook(name)))
                handles.append(module.register_full_backward_hook(backward_hook(name)))

        # we need grads-> run forward+backward on batch
        self.model.train()
        counts = 0
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            # use loss that propagates to logits: sum of target logits
            target_logits = out.gather(1, y.view(-1,1)).squeeze()
            target_logits.sum().backward(retain_graph=False)

            # accumulate stats
            for name in acts:
                a = acts[name].detach().cpu()
                ga = grads_of_acts.get(name)
                if ga is not None:
                    ga = ga.cpu()
                # reduce to per-filter/neuron
                if a.ndim == 4:  # conv -> (B, C, H, W)
                    mean_abs = a.abs().mean(dim=(0,2,3))  # per-channel
                    act_abs_sums[name] = act_abs_sums.get(name, 0) + mean_abs
                    if ga is not None:
                        # activation * grad (mean over spatial)
                        ag = (a * ga).abs().mean(dim=(0,2,3))
                        act_sums[name] = act_sums.get(name, 0) + ag
                elif a.ndim == 2:  # linear -> (B, N)
                    mean_abs = a.abs().mean(dim=0)
                    act_abs_sums[name] = act_abs_sums.get(name, 0) + mean_abs
                    if ga is not None:
                        ag = (a * ga).abs().mean(dim=0)
                        act_sums[name] = act_sums.get(name, 0) + ag
            counts += 1

        # average
        out = {}
        for name in act_abs_sums:
            out[name] = {'mean_abs_activation': act_abs_sums[name] / max(1, counts)}
            if name in act_sums:
                out[name]['mean_abs_act_times_grad'] = act_sums[name] / max(1, counts)

        # remove hooks
        for h in handles:
            h.remove()

        return out

    # -------------------------
    # 6) Grad-CAM style per-filter saliency (simple)
    # -------------------------
    def gradcam_like_filter_saliency(self, dataloader, criterion, num_batches=1):
        """
        For conv layers: compute average over batches of global-pooled (activation * grad_activation) per channel.
        """
        self.model.train()
        activations = {}
        grad_acts = {}
        handles = []

        def f_hook(name):
            def hook(m, i, o):
                activations[name] = o
            return hook

        def b_hook(name):
            def hook(m, grad_i, grad_o):
                grad_acts[name] = grad_o[0]
            return hook

        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(f_hook(n)))
                handles.append(m.register_full_backward_hook(b_hook(n)))

        accum = defaultdict(float)
        counts = 0
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for name in activations:
                a = activations[name].detach()
                ga = grad_acts.get(name)
                if ga is None:
                    continue
                # per-channel pooled saliency
                # (B, C, H, W) -> channel importance: mean over spatial and batch of |a * grad_a|
                sal = (a * ga).abs().mean(dim=(0,2,3)).cpu()
                accum[name] = accum.get(name, 0) + sal
            counts += 1

        for h in handles:
            h.remove()

        for name in list(accum.keys()):
            accum[name] = accum[name] / max(1, counts)
        return dict(accum)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Quick demo on a small model/dataset
    import torchvision
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST('.', train=True, download=True, transform=transform)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)

    estimator = ImportanceEstimator(model, device=device)
    criterion = nn.CrossEntropyLoss()

    # Gradient sensitivity (1 batch)
    grad_info = estimator.gradient_sensitivity(dl, criterion, num_batches=1)
    print("Gradient sensitivity keys:", list(grad_info.keys()))

    # Empirical Fisher (1 batch)
    fisher = estimator.empirical_fisher(dl, criterion, num_batches=1)
    print("Fisher keys:", list(fisher.keys()))

    # Hessian diag (Hutchinson) - use small num_samples
    hessian = estimator.hessian_diag_hutchinson(dl, criterion, num_batches=1, num_samples=3)
    print("Hessian diag keys:", list(hessian['hessian_diag'].keys())[:3])

    # Activation-based metrics
    act_metrics = estimator.activation_based_metrics(dl, num_batches=1)
    print("Activation metric keys:", list(act_metrics.keys()))

    # GradCAM like per-filter saliency
    gradcam = estimator.gradcam_like_filter_saliency(dl, criterion, num_batches=1)
    print("GradCAM-like conv keys:", list(gradcam.keys())[:5])

    # LRP (simplified)
    lrp = estimator.lrp_epsilon(dl, num_batches=1)
    print("LRP keys:", list(lrp.keys())[:5])
