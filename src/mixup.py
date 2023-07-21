from timm.data.mixup import Mixup, mixup_target


class MyMixup(Mixup):
    def __call__(self, x, target):
        assert len(x) % 2 == 0, "Batch size should be even when using this"
        if self.mixup_alpha > 0 or self.cutmix_alpha > 0:
            if self.mode == "elem":
                lam = self._mix_elem(x)
            elif self.mode == "pair":
                lam = self._mix_pair(x)
            else:
                lam = self._mix_batch(x)
        else:
            lam = 1.0
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target
