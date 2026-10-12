import torch
from enrc.rotationLayer import RotationLayer
from ourUtils.clusteringAlgos.nr_init_v3 import init_strategy
# from ourUtils.clusteringAlgos.nr_init_v3 import init_kmeanspp_and_V_random
import numpy as np
from ourUtils.torch import int_to_one_hot


class Enrc(torch.nn.Module):
    def __init__(self, org_space_data, centers, P, V, embedding_fn, random_state):
        super().__init__()
        n_dims = centers[0].shape[1]
        self.init_P = P
        self.beta_weights = self.beta_weights_init(self.init_P, n_dims)
        self.beta_weights = torch.nn.Parameter(self.beta_weights, requires_grad=True)
        self.rotation_layer = RotationLayer(V).to(org_space_data.device)

        self.org_space_data = org_space_data
        self.embedding_fn = embedding_fn
        self.center_lr = None

        self.centers = [torch.tensor(centers_sub, dtype=torch.float32).to(
            org_space_data.device) for centers_sub in centers]
        self.lonely_centers_count = []
        self.mask_sum = []
        for centers_i in self.centers:
            self.lonely_centers_count.append(
                np.zeros((centers_i.shape[0], 1)).astype(int))
            self.mask_sum.append(
                torch.zeros((centers_i.shape[0], 1)))
        self.reinit_threshold = 1

    @staticmethod
    def beta_weights_init(P, n_dims, high_value=0.9, weight_high=1.0):
        n_sub_clusterings = len(P)
        beta_hard = np.zeros((n_sub_clusterings, n_dims), dtype=np.float32)
        for sub_i, p in enumerate(P):
            for dim in p:
                beta_hard[sub_i, dim] = 1.0
        low_value = 1.0 - high_value
        weight_high_exp = np.exp(weight_high)
        # Because high_value = weight_high/(weight_high +low_classes*weight_low)
        n_low_classes = len(P)-1
        weight_low_exp = weight_high_exp * \
            (1.0-high_value)/(high_value*n_low_classes)
        weight_low = np.log(weight_low_exp)
        beta_soft_weights = beta_hard*(weight_high-weight_low) + weight_low
        return torch.tensor(beta_soft_weights, dtype=torch.float32)

    @staticmethod
    def run_init(org_space_data, clusterings, embedding_fn, rounds=100, random_state=1, max_iter=10):
        # scales less good to large data but more stable
        embedded = embedding_fn(org_space_data)
        centers, P, V = init_strategy(data=embedded, clusterings=clusterings,
                                      rounds=rounds, max_iter=max_iter,
                                      random_state=random_state)
        return Enrc(org_space_data, centers, P, V, embedding_fn, random_state)
    
    # @staticmethod
    # def init_with_gradient_descent(org_space_data, trainloader, clustered_spaces, embedding_fn, cluster_lr, rounds=1, max_iter=2000):
    #     # scales better to large data but less stable
    #     return init_with_gradient_descent(org_space_data, trainloader, clustered_spaces, embedding_fn, cluster_lr, rounds=rounds, max_iter=max_iter)
    
    # @staticmethod
    # def run_random_init(org_space_data, clusterings, embedding_fn, rounds=10, random_state=1):
    #     # random initialization of V and centers (with kmeans++ init)
    #     best = None
    #     random_state = check_random_state(random_state)
    #     embedded = embedding_fn(org_space_data).detach().cpu().numpy()
    #     for i in range(rounds):
    #         res = init_kmeanspp_and_V_random(X=embedded, n_clusters=clusterings,  random_state=random_state.randint(10000))
    #         if best is None or best[4] > res[4]:
    #             best = res
    #         print(f"Round {i}: Found solution with: {res[4]} (current best: {best[4]})")
    #     return Enrc(org_space_data, best[3], best[2], best[0], embedding_fn, random_state)

    def subspace_betas(self):
        return torch.nn.functional.softmax(self.beta_weights, dim=0)

    def reinit_centers(self, subspace_id, n_samples=1000, kmeans_steps=10):
        changed = False
        centers = self.centers[subspace_id]
        cluster_space_weights = self.subspace_betas()[subspace_id, :].unsqueeze(0).unsqueeze(1)

        def calculate_sqared_diff_batchwise_and_rotated_z(z, center_id, rotate: bool):
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(z, z)),
                                                      batch_size=64,
                                                      shuffle=False,
                                                      drop_last=False)
            selected_diffs = []
            encodings = []
            for batch in data_loader:
                if rotate:
                    batch_data = batch[0].to(self.rotation_layer.v.device)
                    z_rot = self.rotation_layer.rotate(batch_data)
                    encodings.append(z_rot.detach().cpu())
                else:
                    z_rot = batch[0].to(self.rotation_layer.v.device)

                idx_other_centers = [
                    i for i in range(centers.shape[0]) if i != center_id]
                center_data_diffs = (
                    centers[idx_other_centers].unsqueeze(0) - z_rot.unsqueeze(1))
                weighted_diffs = center_data_diffs * cluster_space_weights
                selected_diff_i = weighted_diffs.pow(2.0).sum(2)
                selected_diffs.append(selected_diff_i)
                torch.cuda.empty_cache()
            if rotate:
                return torch.cat(selected_diffs, 0), torch.cat(encodings, 0)
            else:
                return torch.cat(selected_diffs, 0)

        def rotated_batchwise(z):
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(z, z)),
                                                      batch_size=64,
                                                      shuffle=False,
                                                      drop_last=False)
            encodings = []
            for batch in data_loader:
                batch_data = batch[0].to(self.rotation_layer.v.device)
                z_rot = self.rotation_layer.rotate(batch_data)
                encodings.append(z_rot.detach().cpu())
            return torch.cat(encodings, 0)

        def calculate_batch_cluster_sums_batchwise(z, one_hot_mask):
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(z, z)),
                                                      batch_size=64,
                                                      shuffle=False,
                                                      drop_last=False)
            batch_cluster_sums = None
            from_idx = 0
            to_idx = 0
            for batch in data_loader:
                batch_data = batch[0].to(self.rotation_layer.v.device)
                to_idx += batch_data.shape[0]
                mask_i = one_hot_mask[from_idx:to_idx]
                from_idx = to_idx
                if batch_cluster_sums is None:
                    batch_cluster_sums = (batch_data.unsqueeze(
                        1) * mask_i.unsqueeze(2)).sum(0)
                else:
                    batch_cluster_sums += (batch_data.unsqueeze(1)
                                           * mask_i.unsqueeze(2)).sum(0)
            return batch_cluster_sums

        with torch.no_grad():
            for center_id, count_i in enumerate(self.lonely_centers_count[subspace_id].flatten()):
                if count_i >= self.reinit_threshold:
                    changed = True
                    n_data = self.org_space_data.shape[0]
                    if n_samples > n_data:
                        n_samples = n_data
                    rand_idx = np.random.choice(
                        n_data, size=n_samples, replace=False)
                    x_sampled = self.org_space_data[rand_idx]
                    z_sampled = self.embedding_fn(x_sampled)
                    selected_diff, z_sampled = calculate_sqared_diff_batchwise_and_rotated_z(
                        z_sampled, center_id, rotate=True)

                    kmeans_loss = selected_diff.sum(0)
                    costly_centroid_idx = kmeans_loss.argmax()
                    max_idx = selected_diff[costly_centroid_idx, :].argmax()

                    new_center = z_sampled[max_idx]

                    self.centers[subspace_id][center_id, :] = new_center.to(
                        self.rotation_layer.v.device)
                    # number of kmeans steps
                    for step_i in range(kmeans_steps):
                        centers = self.centers[subspace_id]

                        squared_diff = calculate_sqared_diff_batchwise_and_rotated_z(
                            z_sampled, center_id=None, rotate=False)
                        k = centers.shape[0]
                        arg_min = squared_diff.detach().argmin(1)
                        one_hot_mask = int_to_one_hot(arg_min, k)
                        batch_cluster_sums = calculate_batch_cluster_sums_batchwise(
                            z_sampled, one_hot_mask)
                        mask_sum = one_hot_mask.sum(0)
                        nonzero_mask = (mask_sum != 0)
                        self.centers[subspace_id][nonzero_mask] = batch_cluster_sums[nonzero_mask] / \
                            mask_sum[nonzero_mask].unsqueeze(1)
                        # Reset mask_sum
                        self.mask_sum[subspace_id] = mask_sum.unsqueeze(1)
                    del z_sampled
                    torch.cuda.empty_cache()
                    self.lonely_centers_count[subspace_id][center_id] = 0
        return changed

    def update_center(self, unsqueezed_1_data, one_hot_mask, subspace_id, unsqueezed_1_data_augmented=None):
        batch_cluster_sums = (unsqueezed_1_data.detach()
                              * one_hot_mask.unsqueeze(2)).sum(0)
        mask_sum = one_hot_mask.sum(0).unsqueeze(1)
        if (mask_sum == 0).sum().int().item() != 0:
            idx = (mask_sum == 0).nonzero()[:, 0].detach().cpu()
            self.lonely_centers_count[subspace_id][idx] += 1

        # In case mask sum is zero batch cluster sum is also zero so we can add a small constant to mask sum and center_lr
        # Avoid division by a small number
        mask_sum += 1e-16
        # Use exponential weighted average
        nonzero_mask = (mask_sum.squeeze(1) != 0)
        self.mask_sum[subspace_id][nonzero_mask] = self.center_lr * mask_sum[nonzero_mask] + \
            (1 - self.center_lr) * self.mask_sum[subspace_id][nonzero_mask]
        if (self.mask_sum[subspace_id] == 0).sum().int().item() != 0:
            print(
                f"running mask sum contains zero entry in subspace {subspace_id}:\n {self.mask_sum[subspace_id]} ")
            print("Start reinit")
            self.reinit_centers(
                subspace_id, n_samples=unsqueezed_1_data.shape[0]*10, kmeans_steps=10)

        per_center_lr = 1.0 / (self.mask_sum[subspace_id][nonzero_mask] + 1)
        self.centers[subspace_id] = (
            1.0 - per_center_lr) * self.centers[subspace_id][nonzero_mask] + per_center_lr * batch_cluster_sums[nonzero_mask] / mask_sum[nonzero_mask]
        if torch.isnan(self.centers[subspace_id]).sum() > 0:
            raise ValueError(
                f"Found nan values\n self.centers[subspace_id]: {self.centers[subspace_id]}\n per_center_lr: {per_center_lr}\n self.mask_sum[subspace_id]: {self.mask_sum[subspace_id]}\n ")

    def forward(self, z):
        z_rot = self.rotation_layer.rotate(z)
        z_rot_back = self.rotation_layer.rotate_back(z_rot)

        subspace_betas = self.subspace_betas()
        subspace_losses = torch.zeros(1).to(z.device)

        for i, centers_i in enumerate(self.centers):
            center_data_diffs = (
                centers_i.detach().unsqueeze(0) - z_rot.unsqueeze(1))
            cluster_space_weights = subspace_betas[i, :].unsqueeze(
                0).unsqueeze(1)
            weighted_diffs = center_data_diffs * cluster_space_weights
            weighted_squared_diff = weighted_diffs.pow(2.0).mean(2)

            k = centers_i.shape[0]
            arg_min = weighted_squared_diff.detach().argmin(1)
            one_hot_mask = int_to_one_hot(arg_min, k)
            weighted_squared_diff_masked = weighted_squared_diff * one_hot_mask
            subspace_losses += weighted_squared_diff_masked.sum()
            self.update_center(z_rot.detach().unsqueeze(1),
                               one_hot_mask,
                               subspace_id=i,
                               )

        nr_of_subspaces = subspace_betas.shape[0]
        subspace_losses = subspace_losses / nr_of_subspaces
        return subspace_losses, z_rot_back

    def predict(self, z):
        z_rot = self.rotation_layer.rotate(z)
        subspace_betas = self.subspace_betas()
        z_rot_unsqueezed_1 = z_rot.unsqueeze(1)

        labels = []
        for i, centers_i in enumerate(self.centers):
            center_data_diffs = (centers_i.detach().unsqueeze(0)
                                 - z_rot_unsqueezed_1)
            cluster_space_weights = subspace_betas[i, :].unsqueeze(
                0).unsqueeze(1)
            weighted_diffs = center_data_diffs * cluster_space_weights
            weighted_squared_diff = weighted_diffs.pow(2.0).sum(2)
            labels_sub = weighted_squared_diff.argmin(1)
            labels_sub = labels_sub.detach().cpu().numpy()
            labels.append(labels_sub)
        return np.stack(labels).transpose()

    def predict_batchwise(self, model, data_loader, device=torch.device("cpu")):
        model.eval()
        predictions = []
        for batch in data_loader:
            batch_data = batch[0].to(device)
            z = model.encode(batch_data)
            pred_i = self.predict(z)
            predictions.append(pred_i)
            torch.cuda.empty_cache()
        return np.concatenate(predictions)

    def predict_encode_batchwise(self, model, data_loader, device=torch.device("cpu")):
        model.eval()
        predictions = []
        encodings = []
        for batch in data_loader:
            batch_data = batch[0].to(device)
            z = model.encode(batch_data)
            pred_i = self.predict(z)
            predictions.append(pred_i)
            z_rot = self.rotation_layer.rotate(z)
            encodings.append(z_rot.detach().cpu())
            torch.cuda.empty_cache()
        return np.concatenate(predictions), torch.cat(encodings, 0)


# def init_with_gradient_descent(org_space_data, trainloader, clustered_spaces, embedding_fn, cluster_lr, rounds=1, max_iter=2000):
#     # scales better to large data but less stable
#     best_loss = np.inf
#     best_round = -1
#     np_seeds = np.random.randint(1000000, size=rounds)
#     for run_i in range(rounds):
#         print("########################################")
#         print(f"Start init run: {run_i} with seed: {np_seeds[run_i]}")
#         cluster_layer = Enrc.run_random_init(org_space_data=org_space_data,
#                                             clusterings=clustered_spaces,
#                                             embedding_fn=embedding_fn,
#                                             rounds=100,
#                                             perform_rot=False,
#                                             run_kmeans=False,
#                                             random_state=np_seeds[run_i],
#                                             )
#         init_P = cluster_layer.init_P
#         torch.cuda.empty_cache()
#         cluster_layer.cuda()
#         cluster_layer.centers = [i.cuda() for i in cluster_layer.centers]
#         cluster_layer.mask_sum = [i.cuda() for i in cluster_layer.mask_sum]
#         param_dict = [{'params': [cluster_layer.rotation_layer.v], 'lr': cluster_lr / 2},
#                         {'params': [cluster_layer.beta_weights],
#                         'lr': cluster_lr}
#                         ]
#         optimizer = torch.optim.Adam(param_dict)
#         cluster_layer.center_lr = 0.5
#         i = 0
#         run = True
#         subspace_losses_list = []
#         while(run):
#             for batch in trainloader:
#                 batch = batch[0].cuda()
#                 z = embedding_fn(batch)
#                 subspace_losses, z_rot_back = cluster_layer(z)
#                 rotation_loss = cluster_layer.rotation_layer.rotation_layer_loss(
#                     z, z_rot_back)
#                 summed_loss = subspace_losses + rotation_loss
#                 optimizer.zero_grad()
#                 summed_loss.backward()
#                 optimizer.step()
#                 subspace_losses = subspace_losses.item()
#                 if i == 0:
#                     print(
#                         f"Steps {i} summed_loss: {summed_loss.item():.4f}, subspace_losses: {subspace_losses:.4f},rotation_loss: {rotation_loss.item():.4f}")
#                 i += 1
#                 subspace_losses_list.append(summed_loss.item())
#                 if i == max_iter:
#                     run = False
#                     break

#         print("After iteration: ", i)
#         with torch.no_grad():
#             print(f"Steps {i} summed_loss: {summed_loss.item():.4f}, subspace_losses: {subspace_losses:.4f}, rotation_loss: {rotation_loss.item():.4f}")

#         subspace_losses = np.min(subspace_losses_list)
#         if subspace_losses < best_loss:
#             best_loss = subspace_losses
#             cluster_layer.centers = [i.detach().cpu().numpy()
#                                         for i in cluster_layer.centers]
#             best_centers = deepcopy(cluster_layer.centers)
#             best_V = cluster_layer.rotation_layer.v.detach().cpu().numpy()
#             best_betas = cluster_layer.beta_weights.detach().cpu()
#             best_round = run_i
#             best_optim_state_dict = deepcopy(optimizer.state_dict())
#         print(f"Round {run_i}: Found solution with: {subspace_losses} (current best round {best_round} with {best_loss})")
#         print("########################################")
#         del cluster_layer
#         torch.cuda.empty_cache()
#     best_cluster_layer = Enrc(org_space_data=org_space_data,
#                                 centers=best_centers,
#                                 # dummy P
#                                 P=init_P,
#                                 V=best_V,
#                                 embedding_fn=embedding_fn,
#                                 random_state=None)
#     # Set beta_weights to best_betas
#     best_cluster_layer.beta_weights = torch.nn.Parameter(best_betas, requires_grad=True)
#     return best_cluster_layer
