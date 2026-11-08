"""Euclidean horizontal lifts for all V/O attention heads with fixed V biases."""

import math

import torch


class VOQuotient:
    def __init__(self, model, found, anchor, head_width=64):
        self.width = head_width
        offset = 0
        slices = {}
        for name, parameter in found:
            slices[name] = (slice(offset, offset + parameter.numel()), parameter.shape)
            offset += parameter.numel()
        self.layers = []
        bias = []
        parameters = dict(model.named_parameters())
        for name, (location, shape) in slices.items():
            if not name.endswith('attn.in_proj_weight'):
                continue
            prefix = name.removesuffix('in_proj_weight')
            output, _ = slices[prefix + 'out_proj.weight']
            bias_name = 'image_encoder.' + prefix + 'in_proj_bias'
            bias.append(parameters[bias_name].detach()[2 * shape[1]:].view(-1, head_width))
            self.layers.append((prefix, location, output, shape[1]))
        self.bias = torch.cat(bias)
        self.value, self.output = self.pack(anchor)
        # Restrict H to H b = 0. The first column of Q is zero for nonzero b.
        q = torch.linalg.qr(self.bias.double().unsqueeze(-1), mode='complete').Q
        fixed = self.bias.norm(dim=-1) > 0
        q[fixed, :, 0] = 0
        vd, od = self.value.double(), self.output.double()
        a, left = torch.linalg.eigh(od.mT @ od)
        b, right = torch.linalg.eigh(q.mT @ (vd @ vd.mT) @ q)
        self.left = left.float()
        self.right = (q @ right).float()
        self.denominator = (a.unsqueeze(-1) + b.unsqueeze(-2)).clamp_min(1e-12).float()
        self.output_min_eig = float(a.min())
        self.vertical_dimension = len(fixed) * head_width ** 2 - int(fixed.sum()) * head_width

    def pack(self, vector):
        values, outputs = [], []
        for _, v, o, d in self.layers:
            values.append(vector[v].view(3 * d, d)[2 * d:].view(-1, self.width, d))
            outputs.append(vector[o].view(d, -1, self.width).permute(1, 0, 2))
        return torch.cat(values), torch.cat(outputs)

    def unpack_(self, vector, values, outputs):
        at = 0
        for _, v, o, d in self.layers:
            end = at + d // self.width
            vector[v].view(3 * d, d)[2 * d:].copy_(values[at:end].reshape(d, d))
            vector[o].view(d, d).copy_(outputs[at:end].permute(1, 0, 2).reshape(d, d))
            at = end
        return vector

    def horizontal(self, vector):
        dv, do = self.pack(vector)
        rhs = dv @ self.value.mT - self.output.mT @ do
        h = self.left @ ((self.left.mT @ rhs @ self.right) / self.denominator) @ self.right.mT
        return self.unpack_(vector.clone(), dv - h @ self.value, do + self.output @ h)

    def check_projection(self, vector):
        projected = self.horizontal(vector)
        vertical = vector - projected
        again = self.horizontal(projected)
        size = vector.norm().clamp_min(1e-12)
        return dict(idempotence=float((again - projected).norm() / size),
                    orthogonality=float(torch.dot(projected, vertical).abs() / size.square()),
                    removed_energy=float((vertical.norm() / size).square()))

    @torch.no_grad()
    def verify_group(self, model, found, anchor, images, every_head):
        """Finite GL changes, including the V bias. Test heads separately and jointly."""
        names = ['image_encoder.' + name for name, _ in found]
        sizes = [parameter.numel() for _, parameter in found]
        shapes = [parameter.shape for _, parameter in found]
        params = {name: value.view(shape) for name, value, shape in zip(names, anchor.split(sizes), shapes)}
        bias_names = ['image_encoder.' + prefix + 'in_proj_bias' for prefix, _, _, _ in self.layers]
        biases = [dict(model.named_parameters())[n].detach() for n in bias_names]
        count = len(self.bias)
        eye = torch.eye(self.width, device=anchor.device, dtype=torch.double)
        gen = torch.Generator(device=anchor.device).manual_seed(17)
        sample = torch.randn(count, self.width, self.width, generator=gen, device=anchor.device, dtype=torch.double)
        transform = eye + .12 / math.sqrt(self.width) * sample
        inverse = torch.linalg.inv(transform)
        v = (transform @ self.value.double()).float()
        o = (self.output.double() @ inverse).float()
        b = (transform @ self.bias.double().unsqueeze(-1)).squeeze(-1).float()
        joint = self.unpack_(anchor.clone(), v, o)
        joint_params = {name: value.view(shape) for name, value, shape in zip(names, joint.split(sizes), shapes)}
        baseline = torch.func.functional_call(model, params, (images,)).double()
        single = torch.func.functional_call(model, params, (images[:1],)).double() if every_head else None
        at = 0
        head_errors = []
        for (prefix, _, _, d), bn, bias in zip(self.layers, bias_names, biases):
            heads = d // self.width
            new_bias = bias.clone()
            new_bias[2 * d:] = b[at:at + heads].reshape(-1)
            joint_params[bn] = new_bias
            if every_head:
                vn = 'image_encoder.' + prefix + 'in_proj_weight'
                on = 'image_encoder.' + prefix + 'out_proj.weight'
                for i in range(heads):
                    index = slice(i * self.width, (i + 1) * self.width)
                    value_index = slice(2 * d + index.start, 2 * d + index.stop)
                    vv, oo, bb = params[vn].clone(), params[on].clone(), bias.clone()
                    vv[value_index], oo[:, index], bb[value_index] = v[at + i], o[at + i], b[at + i]
                    changed = dict(params, **{vn: vv, on: oo, bn: bb})
                    pred = torch.func.functional_call(model, changed, (images[:1],)).double()
                    head_errors.append(float((pred - single).abs().max()))
            at += heads
        together = torch.func.functional_call(model, joint_params, (images,)).double()
        first_v = 'image_encoder.' + self.layers[0][0] + 'in_proj_weight'
        broken = torch.func.functional_call(model, dict(params, **{first_v: joint_params[first_v]}), (images,)).double()
        return dict(heads=count, head_trials=len(head_errors), head_errors=head_errors,
                    max_head_logit_error=max(head_errors, default=None),
                    joint_logit_error=float((together - baseline).abs().max()),
                    broken_logit_error=float((broken - baseline).abs().max()),
                    group_dimension=count * self.width ** 2, fixed_bias_dimension=self.vertical_dimension,
                    minimum_output_gram_eigenvalue=self.output_min_eig)
