from model import unet

if __name__ == "__main__":
    model = unet(512, 1, 3)
    model.summary()