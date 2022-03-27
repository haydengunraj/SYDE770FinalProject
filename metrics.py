import torch


class Metric:
    """Base class for all metrics"""
    def __init__(self, name, log_interval):
        self.name = name
        self.log_interval = log_interval

    def update(self, data_dict):
        """Update running values"""
        raise NotImplementedError

    def reset(self):
        """Reset running values"""
        raise NotImplementedError

    def log(self, writer, step, tag_prefix='val/'):
        """Log the current value(s)"""
        raise NotImplementedError

    @property
    def value(self):
        """Compute final metric value"""
        raise NotImplementedError


class LogMetric(Metric):
    @property
    def value(self):
        """Compute final metric value"""
        return None


class LossMetric(Metric):
    """Running loss metric"""
    def __init__(self, name, log_interval, loss_key='loss'):
        super().__init__(name, log_interval)
        self.loss_key = loss_key
        self.loss_sum = 0
        self.loss_count = 0

    def update(self, data_dict):
        self.loss_sum += data_dict[self.loss_key].item()
        self.loss_count += 1

    def reset(self):
        self.loss_sum = 0
        self.loss_count = 0

    def log(self, writer, step, tag_prefix='val/'):
        writer.add_scalar(tag_prefix + self.name, self.value, step)

    @property
    def value(self):
        return self.loss_sum/self.loss_count


class ImageLogMetric(LogMetric):
    """Stores images for logging purposes"""
    def __init__(self, name, log_interval, image_key='images', max_images=10, mean=None, std=None):
        super().__init__(name, log_interval)
        self.image_key = image_key
        self.max_images = max_images
        self.mean = mean
        self.std = std
        self.images = []

    def update(self, data_dict):
        if len(self.images) < self.max_images:
            image = self.denormalize(data_dict[self.image_key][0])
            self.images.append(image)

    def reset(self):
        self.images = []

    def log(self, writer, step, tag_prefix='val/'):
        for i, image in enumerate(self.images):
            writer.add_image('{}{}_{}'.format(tag_prefix, self.name, i), image, step)

    def denormalize(self, image):
        if self.std is not None:
            std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)
            image = image*std[:, None, None]
        if self.mean is not None:
            mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)
            image = image + mean[:, None, None]
        return image


class OverlayLogMetric(ImageLogMetric):
    """Stores images with overlays for logging purposes"""
    def __init__(self, name, log_interval, image_key='images', mask_key='mask',
                 colour=(1, 0, 0), alpha=0.3, max_images=10, mean=None, std=None):
        super().__init__(name, log_interval, image_key=image_key, max_images=max_images, mean=mean, std=std)
        self.mask_key = mask_key
        self.colour = colour
        self.alpha = alpha

    def update(self, data_dict):
        if len(self.images) < self.max_images:
            image = self.denormalize(data_dict[self.image_key][0])
            mask = data_dict[self.mask_key][0]
            colour = torch.ones((3,) + tuple(image.size()[1:]), device=image.device)
            for i, col in enumerate(self.colour):
                colour[i, ...] *= col
            alpha_mask = mask*self.alpha
            overlay = alpha_mask*colour + (1. - alpha_mask)*image
            self.images.append(overlay)


def update_and_log_metrics(epoch, step, data_dict, metrics, writer, tag_prefix='train/'):
    """Update metrics for the given step"""
    log_str = '[epoch: {}, step: {}'.format(epoch, step)
    print_ = False
    for metric in metrics:
        metric.update(data_dict)
        if not step % metric.log_interval:
            if writer is not None:
                metric.log(writer, step, tag_prefix)
            value = metric.value
            if value is not None:
                log_str += ', {}: {:.3f}'.format(metric.name, value)
                print_ = True
            metric.reset()
    if print_:
        print(log_str + ']', flush=True)
