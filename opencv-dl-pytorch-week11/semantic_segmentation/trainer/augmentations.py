import random

from albumentations import ImageOnlyTransform


class Cutout(ImageOnlyTransform):
    def __init__(self, img_size, rect_size, fill_value=(0, 0, 0), always_apply=False, p=1.0):
        super(Cutout, self).__init__(always_apply, p)
        self.img_size = img_size
        self.rect_size = rect_size
        self.fill_value = fill_value

    def apply(self, img, x, y, **params):
        img[y:y + self.rect_size[1], x:x + self.rect_size[1], :] = self.fill_value
        return img

    def get_params_dependent_on_targets(self, params):
        pass

    def get_params(self):
        x = int(random.uniform(0, self.img_size[1] - self.rect_size[1]))
        y = int(random.uniform(0, self.img_size[0] - self.rect_size[0]))
        return {"x": x, "y": y}

    def get_transform_init_args_names(self):
        return ("img_size", "rect_size", "fill_value")
