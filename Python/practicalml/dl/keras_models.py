from keras.applications import VGG16

class ModelRepository(object):
    def get_vgg16(self):
        path = 'D:\code\ML\models\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg16 = VGG16(weights='imagenet', weights_path=path, include_top=False, input_shape=(150, 150, 3))
        return vgg16