import math

def pixel_to_3d(x, y, depth, fx, fy, cx, cy):
    """
    Converts 2D pixel coordinates (x, y) and depth (Z) into 3D real-world coordinates (X, Y, Z).
    
    Args:
        x (float): x-coordinate of the pixel.
        y (float): y-coordinate of the pixel.
        depth (float): depth value at (x, y) - this is Z.
        fx (float): focal length in x.
        fy (float): focal length in y.
        cx (float): principal point x (usually width / 2).
        cy (float): principal point y (usually height / 2).
        
    Returns:
        tuple: (X, Y, Z) 3D coordinates.
    """
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return (X, Y, Z)

def calculate_3d_distance(point1, point2):
    """
    Calculates the Euclidean distance between two 3D points.
    
    Args:
        point1 (tuple): (X1, Y1, Z1)
        point2 (tuple): (X2, Y2, Z2)
        
    Returns:
        float: 3D Euclidean distance.
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def estimate_intrinsics(image_width, image_height, fov_degrees=60.0):
    """
    Estimates camera intrinsics if they are not known.
    Assumes the principal point is at the image center and a default field of view.
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # fx = fy = (width / 2) / tan(FOV / 2)
    fov_radians = math.radians(fov_degrees)
    f = (image_width / 2.0) / math.tan(fov_radians / 2.0)
    
    return f, f, cx, cy
