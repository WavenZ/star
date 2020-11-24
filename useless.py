import matplotlib.pyplot as plt
import generator.generate_starImg as ggs

if __name__ == "__main__":
    
    # Attitude: (yaw, pitch, roll)
    att = [20, 40, 60]

    # Angle Velocity: (yaw, pitch, roll)
    dps = [5, 5, 5]

    # Generate
    retImg = ggs.genDynamic(att, dps, 100)

    # Show
    plt.figure()
    plt.imshow(retImg, cmap='gray', vmin=0, vmax=255)
    plt.show()
