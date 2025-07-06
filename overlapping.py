import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon # Renamed to avoid conflict

def sutherland_hodgman_original(subject_polygon, clip_polygon):
    """
    Clips the subject_polygon against the clip_polygon using the Sutherland-Hodgman algorithm.

    Args:
        subject_polygon (list of tuples): List of (x, y) coordinates representing the subject polygon.
        clip_polygon (list of tuples): List of (x, y) coordinates representing the clip polygon.

    Returns:
        list of tuples: List of (x, y) coordinates representing the intersection polygon.
                        Returns an empty list if there is no intersection.
    """
    output_polygon = list(subject_polygon) # Start with the subject polygon
    for i in range(len(clip_polygon)):
        clip_edge_start = clip_polygon[i]
        clip_edge_end = clip_polygon[(i + 1) % len(clip_polygon)]

        new_output_polygon = []
        if not output_polygon: # If the subject polygon became empty in a previous step, no intersection
            break

        for j in range(len(output_polygon)):
            subject_edge_start = output_polygon[j]
            subject_edge_end = output_polygon[(j + 1) % len(output_polygon)]

            # Check if points are inside/outside the clip edge
            # A point (px, py) is "inside" a directed edge (cx1, cy1) -> (cx2, cy2)
            # if (cx2 - cx1)(py - cy1) - (cy2 - cy1)(px - cx1) is negative (or zero for on the line)
            # This is the cross product / z-component of cross product (vector from start to point) x (vector from start to end)
            is_start_inside = is_inside(subject_edge_start, clip_edge_start, clip_edge_end)
            is_end_inside = is_inside(subject_edge_end, clip_edge_start, clip_edge_end)

            if is_start_inside and is_end_inside:
                # Case 1: Both points are inside, add the end point
                new_output_polygon.append(subject_edge_end)
            elif is_start_inside and not is_end_inside:
                # Case 2: Start is inside, end is outside, add intersection point
                intersection_point = compute_intersection_original(subject_edge_start, subject_edge_end,
                                                          clip_edge_start, clip_edge_end)
                new_output_polygon.append(intersection_point)
            elif not is_start_inside and is_end_inside:
                # Case 3: Start is outside, end is inside, add intersection point and then the end point
                intersection_point = compute_intersection_original(subject_edge_start, subject_edge_end,
                                                          clip_edge_start, clip_edge_end)
                new_output_polygon.append(intersection_point)
                new_output_polygon.append(subject_edge_end)
            # Case 4: Both points are outside, do nothing

        output_polygon = new_output_polygon

    return output_polygon

def is_inside(point, clip_edge_start, clip_edge_end):
    """
    Checks if a point is inside (or on the boundary) of the half-plane defined by the clip edge.
    Assuming clip polygon vertices are ordered such that "inside" is to the left of the directed edge.
    (cx2 - cx1)(py - cy1) - (cy2 - cy1)(px - cx1) < 0  (for left side)
    """
    cx1, cy1 = clip_edge_start
    cx2, cy2 = clip_edge_end
    px, py = point
    return ((cx2 - cx1) * (py - cy1) - (cy2 - cy1) * (px - cx1)) >= 0

def compute_intersection_original(p1, p2, clip_p1, clip_p2):
    """
    Computes the intersection point of two line segments (p1, p2) and (clip_p1, clip_p2).
    Assumes an intersection exists.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = clip_p1
    x4, y4 = clip_p2

    # Line A: p1 -> p2 (x1, y1) to (x2, y2)
    # Line B: clip_p1 -> clip_p2 (x3, y3) to (x4, y4)

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0: # Ideally: if den == 0:
        # Lines are parallel or collinear. For convex polygons, this means
        # either no intersection or overlap, but for Sutherland-Hodgman,
        # we expect a distinct intersection point when crossing the boundary.
        # This case should ideally not be hit for proper intersections in this algorithm.
        # For simplicity in this context, we'll return one of the endpoints as a fallback,
        # but in a robust system, you'd handle collinear cases more carefully.
        return p1 # Or raise an error
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    # u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den # Not strictly needed for intersection point

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    return (ix, iy)

def visualize_polygons(poly1, poly2, title="Polygon Intersection"):
    """
    Visualizes two polygons and their intersection.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Plot Polygon 1
    if poly1:
        mpl_poly1 = MplPolygon(poly1, closed=True, edgecolor='blue', facecolor='blue', alpha=0.3, label='Polygon 1')
        ax.add_patch(mpl_poly1)
        poly1_np = np.array(poly1)
        ax.plot(poly1_np[:, 0], poly1_np[:, 1], 'o', color='blue', markersize=5) # Vertices

    # Plot Polygon 2
    if poly2:
        mpl_poly2 = MplPolygon(poly2, closed=True, edgecolor='red', facecolor='red', alpha=0.3, label='Polygon 2')
        ax.add_patch(mpl_poly2)
        poly2_np = np.array(poly2)
        ax.plot(poly2_np[:, 0], poly2_np[:, 1], 'o', color='red', markersize=5) # Vertices

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()

def visualize_polygons_with_intersection(poly1, poly2, intersection_poly=None, title="Polygon Intersection"):
    """
    Visualizes two polygons and their intersection.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot Polygon 1
    if poly1:
        mpl_poly1 = MplPolygon(poly1, closed=True, edgecolor='blue', facecolor='blue', alpha=0.3, label='Polygon 1')
        ax.add_patch(mpl_poly1)
        poly1_np = np.array(poly1)
        ax.plot(poly1_np[:, 0], poly1_np[:, 1], 'o', color='blue', markersize=5) # Vertices

    # Plot Polygon 2
    if poly2:
        mpl_poly2 = MplPolygon(poly2, closed=True, edgecolor='red', facecolor='red', alpha=0.3, label='Polygon 2')
        ax.add_patch(mpl_poly2)
        poly2_np = np.array(poly2)
        ax.plot(poly2_np[:, 0], poly2_np[:, 1], 'o', color='red', markersize=5) # Vertices

    # Plot Intersection Polygon
    if intersection_poly and len(intersection_poly) > 0:
        mpl_intersection = MplPolygon(intersection_poly, closed=True, edgecolor='green', facecolor='green', alpha=0.7, label='Intersection')
        ax.add_patch(mpl_intersection)
        intersection_np = np.array(intersection_poly)
        ax.plot(intersection_np[:, 0], intersection_np[:, 1], 'x', color='green', markersize=8, markeredgewidth=2) # Intersection vertices

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)
    #ax.set_xlim(0, 1550)
    #ax.set_ylim(2048, 0)
    ax.yaxis.set_inverted(True)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()

def polygon_area(polygon_vertices):
    """
    Calculates the area of a polygon using the shoelace formula.
    The polygon must be simple (not self-intersecting).

    Args:
        polygon_vertices (list of tuples): List of (x, y) coordinates representing the polygon vertices.

    Returns:
        float: The area of the polygon. Returns 0.0 if the polygon has fewer than 3 vertices.
    """
    n = len(polygon_vertices)
    if n < 3: # A polygon needs at least 3 vertices to have an area
        return 0.0

    area = 0.0
    for i in range(n):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % n] # (i + 1) % n wraps around to 0 for the last vertex

        area += (x1 * y2) - (x2 * y1)

    return abs(area / 2.0)

