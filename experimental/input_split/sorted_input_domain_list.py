class InputDomain:
    """Singleton domain for input split, used by sorted domain list."""

    def __init__(
        self,
        lb=-float("inf"),
        ub=float("inf"),
        lb_all=None,
        up_all=None,
        alpha=None,
        dm_l=None,
        dm_u=None,
        selected_dims=None,
        device="cpu",
        depth=None,
        c=None,
        threshold=0,
        split_idx=None,
    ):
        self.lower_bound = lb
        self.upper_bound = ub
        self.dm_l = dm_l
        self.dm_u = dm_u
        self.lower_all = lb_all
        self.upper_all = up_all
        self.alpha = alpha
        self.selected_dims = selected_dims
        self.device = device
        self.split = False
        self.valid = True
        self.depth = depth
        self.lA = None
        self.beta = None
        self.intermediate_betas = None
        self.c = c
        self.threshold = threshold
        self.split_idx = split_idx

    def __lt__(self, other):
        return (self.lower_bound - self.threshold).max() < (
            other.lower_bound - other.threshold
        ).max()

    def __le__(self, other):
        return (self.lower_bound - self.threshold).max() <= (
            other.lower_bound - other.threshold
        ).max()

    def __eq__(self, other):
        return (self.lower_bound - self.threshold).max() == (
            other.lower_bound - other.threshold
        ).max()

    def __repr__(self):
        string = (
            f"LB: {self.lower_bound - self.threshold}\n"
            if self.lower_bound is not None
            else ""
        )
        string += (
            f"UB: {self.upper_bound - self.threshold}\n"
            if self.upper_bound is not None
            else ""
        )
        return string

    def verify_criterion(self):
        return (self.lower_bound > self.threshold).any()

    def to_cpu(self):
        if self.device == "cuda":
            return self
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.dm_l = self.dm_l.to(device="cpu", non_blocking=True)
        self.dm_u = self.dm_u.to(device="cpu", non_blocking=True)
        self.lower_bound = self.lower_bound.to(device="cpu", non_blocking=True)
        self.upper_bound = self.upper_bound.to(device="cpu", non_blocking=True)
        if self.selected_dims is not None:
            self.selected_dims = self.selected_dims.to(device="cpu", non_blocking=True)

        if self.c is not None:
            self.c = self.c.to(device="cpu", non_blocking=True)

        self.threshold = self.threshold.to(device="cpu", non_blocking=True)

        if self.alpha is not None:
            for layer in self.alpha:
                for intermediate_layer in self.alpha[layer]:
                    self.alpha[layer][intermediate_layer] = (
                        self.alpha[layer][intermediate_layer]
                        .half()
                        .to(device="cpu", non_blocking=True)
                    )

        return self

    def to_device(self, device, partial=False):
        if self.device == "cuda":
            return self
        self.dm_l = self.dm_l.to(device, non_blocking=True)
        self.dm_u = self.dm_u.to(device, non_blocking=True)
        self.lower_bound = (
            self.lower_bound.to(device, non_blocking=True)
            if self.lower_bound is not None
            else None
        )

        if self.c is not None:
            self.c = self.c.to(device, non_blocking=True)

        self.threshold = self.threshold.to(device, non_blocking=True)

        if self.alpha is not None:
            for layer in self.alpha:
                for intermediate_layer in self.alpha[layer]:
                    self.alpha[layer][intermediate_layer] = (
                        self.alpha[layer][intermediate_layer]
                        .to(device, non_blocking=True)
                        .to(torch.get_default_dtype())
                    )

        return self


class SortedInputDomainList(InputDomainList):
    """Sorted domain list for input split."""

    def __init__(self):
        super(SortedInputDomainList, self).__init__()
        self.domains = SortedList()
        min_batch_size = (
            arguments.Config["solver"]["beta-crown"]["min_batch_size_ratio"]
            * arguments.Config["solver"]["batch_size"]
        )
        self.max_depth = max(int(math.log(max(min_batch_size, 1)) // math.log(2)), 1)

    def __len__(self):
        return len(self.domains)

    def __getitem__(self, idx):
        domain = self.domains[idx]
        return domain.lower_bound, domain.dm_l, domain.dm_u, domain.c, domain.threshold

    def add(
        self,
        lb,
        dm_l,
        dm_u,
        alpha,
        cs,
        threshold=0,
        split_idx=None,
        remaining_index=None,
    ):
        threshold = threshold.to(device=lb.device)
        dm_l = dm_l.to(device=lb.device)
        dm_u = dm_u.to(device=lb.device)
        cs = cs.to(device=lb.device)
        if remaining_index is None:
            # FIXME: this should check the criterion function.
            remaining_index = torch.where((lb <= threshold).all(1))[0]
        for i in remaining_index:
            if alpha is not None and type(alpha) != list:
                alpha_dict = {}
                for key0 in alpha.keys():
                    alpha_dict[key0] = {}
                    for key1 in alpha[key0].keys():
                        alpha_dict[key0][key1] = alpha[key0][key1][:, :, i : i + 1]
            dom = InputDomain(
                lb=lb[i],
                alpha=alpha_dict if alpha is not None and type(alpha) != list else None,
                dm_l=dm_l[i : i + 1],
                dm_u=dm_u[i : i + 1],
                threshold=threshold[i],
                c=cs[i : i + 1],
                split_idx=(split_idx[i : i + 1] if split_idx is not None else None),
            )
            self.domains.add(dom)

    def pick_out_batch(self, batch, device="cuda"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # make sure GPU to CPU transfer is finished

        batch = min(len(self.domains), batch)
        # Lower and upper bounds of inputs.
        dm_l_all, dm_u_all = [], []
        # Specification matrices, and decision thresholds.
        c_all, thresholds_all = [], []
        alphas_all, split_idx = [], []
        assert len(self.domains) > 0, "The given domains list is empty."

        for i in range(batch):
            # Pop out domains from the list one by one (SLOW).
            selected_candidate_domain = self.domains.pop(0)
            # We transfer only some of the tensors directly to GPU. Other tensors will be transfered in batch later.
            selected_candidate_domain.to_device(device, partial=True)
            alphas_all.append(selected_candidate_domain.alpha)
            dm_l_all.append(selected_candidate_domain.dm_l)
            dm_u_all.append(selected_candidate_domain.dm_u)
            c_all.append(selected_candidate_domain.c)
            thresholds_all.append(selected_candidate_domain.threshold)
            if (
                split_idx is not None
                and selected_candidate_domain.split_idx is not None
            ):
                split_idx.append(selected_candidate_domain.split_idx)
            else:
                split_idx = None

        thresholds = torch.stack(thresholds_all).to(device=device, non_blocking=True)
        split_idx = torch.cat(split_idx) if split_idx is not None else None

        # aggregate C to shape (batch, 1, num_outputs)
        cs = torch.cat(c_all).contiguous().to(device=device, non_blocking=True)

        # Input split domains.
        return (
            alphas_all,
            torch.cat(dm_l_all).to(device=device, non_blocking=True),
            torch.cat(dm_u_all).to(device=device, non_blocking=True),
            cs,
            thresholds,
            split_idx,
        )

    def get_topk_indices(self, k=1, largest=False):
        assert k <= self.__len__(), print("Asked indices more than domain length.")
        return -torch.arange(k) - 1 if largest else torch.arange(k)
