class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same')
        self.layer2 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same')
        self.layer3 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same')
        self.layer4 = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same')
        self.layer5 = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same')
        self.layer6 = tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same')
    # @tf.function
    # def conv2d(layer_input, filters, f_size=4, bn=True):
    #     d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    #     d = LeakyReLU(alpha=0.2)(d)
    #     if bn:
    #         d = BatchNormalization(momentum=0.8)(d)
    #     return d


    def call(self, images):
        gf = 32
        images = self.layer1(images)
        images = LeakyReLU(alpha=0.2)(images)
        images = BatchNormalization(momentum=0.8)(images)
        images = self.layer2(images)
        images = LeakyReLU(alpha=0.2)(images)
        images = self.layer3(images)
        images = LeakyReLU(alpha=0.2)(images)
        images = self.layer4(images)
        images = LeakyReLU(alpha=0.2)(images)
        images = self.layer5(images)
        images = LeakyReLU(alpha=0.2)(images)
        images = self.layer6(images)
        images = LeakyReLU(alpha=0.2)(images)

        return images

def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        return u
class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu')
        self.layer2 = Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')
        self.layer3 = Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu')
        self.layer4 = Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.layer5 = Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.layer6 = Conv2D(3, kernel_size=5, strides=1, padding='same', activation='relu')

    def call(self, encoder_output):
        print(encoder_output.shape)
        image = UpSampling2D(size=2)(encoder_output)
        image = self.layer1(image)
        image = BatchNormalization(momentum=0.8)(image)
        image = UpSampling2D(size=2)(image)
        image = self.layer2(image)
        image = BatchNormalization(momentum=0.8)(image)
        image = UpSampling2D(size=2)(image)
        image = self.layer3(image)
        image = BatchNormalization(momentum=0.8)(image)
        image = UpSampling2D(size=2)(image)
        image = self.layer4(image)
        image = BatchNormalization(momentum=0.8)(image)
        image = UpSampling2D(size=2)(image)
        image = self.layer5(image)
        image = BatchNormalization(momentum=0.8)(image)
        image = UpSampling2D(size=2)(image)
        output_img = self.layer6(image)
        print(output_img.shape)
        return output_img