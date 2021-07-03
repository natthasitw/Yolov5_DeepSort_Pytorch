from collections import deque
import numpy as np
import cv2
from random import randint


class Zone:
    '''
    coord are stored as list of (x,y)
    '''
    def __init__(self, coord, zoneID):
        self.zoneID = zoneID
        self.coord = coord
        self.n_static = 0
        self.n_tz = 0
        self.tzIDs = deque(maxlen=int(50))
        self.tzfeat = deque(maxlen=int(50))

    def _reset(self):
        self.n_static = 0
        self.n_tz = 0

    def _add_ID(self, new_id, new_feat):
        if new_id not in self.tzIDs:
            self.tzIDs.append(new_id)
            self.tzfeat.append(new_feat)
            self.n_static += 1


    def is_inside_polygon(self, p):
        '''

        Parameters
        ----------
        p is a tuple of (x,y)

        Returns
        -------

        '''

        def _orientation(p: tuple, q: tuple, r: tuple) -> int:

            val = (((q[1] - p[1]) *
                    (r[0] - q[0])) -
                   ((q[0] - p[0]) *
                    (r[1] - q[1])))

            if val == 0:
                return 0
            if val > 0:
                return 1  # Collinear
            else:
                return 2  # Clock or counterclock

        def _onSegment(p: tuple, q: tuple, r: tuple) -> bool:

            if ((q[0] <= max(p[0], r[0])) &
                    (q[0] >= min(p[0], r[0])) &
                    (q[1] <= max(p[1], r[1])) &
                    (q[1] >= min(p[1], r[1]))):
                return True

            return False

        def _doIntersect(p1, q1, p2, q2):

            # Find the four orientations needed for
            # general and special cases
            o1 = _orientation(p1, q1, p2)
            o2 = _orientation(p1, q1, q2)
            o3 = _orientation(p2, q2, p1)
            o4 = _orientation(p2, q2, q1)

            # General case
            if (o1 != o2) and (o3 != o4):
                return True

            # Special Cases
            # p1, q1 and p2 are colinear and
            # p2 lies on segment p1q1
            if (o1 == 0) and (_onSegment(p1, p2, q1)):
                return True

            # p1, q1 and p2 are colinear and
            # q2 lies on segment p1q1
            if (o2 == 0) and (_onSegment(p1, q2, q1)):
                return True

            # p2, q2 and p1 are colinear and
            # p1 lies on segment p2q2
            if (o3 == 0) and (_onSegment(p2, p1, q2)):
                return True

            # p2, q2 and q1 are colinear and
            # q1 lies on segment p2q2
            if (o4 == 0) and (_onSegment(p2, q1, q2)):
                return True

            return False

        n = len(self.coord)

        # There must be at least 3 vertices
        # in polygon
        if n < 3:
            return False

        # Create a point for line segment
        # from p to infinite
        extreme = (10000, p[1])
        count = i = 0

        while True:
            next = (i + 1) % n

            # Check if the line segment from 'p' to
            # 'extreme' intersects with the line
            # segment from 'polygon[i]' to 'polygon[next]'
            if (_doIntersect(self.coord[i],
                            self.coord[next],
                            p, extreme)):

                # If the point 'p' is colinear with line
                # segment 'i-next', then check if it lies
                # on segment. If it lies, return true, otherwise false
                if _orientation(self.coord[i], p,
                               self.coord[next]) == 0:
                    return _onSegment(self.coord[i], p,
                                     self.coord[next])

                count += 1

            i = next

            if (i == 0):
                break

        # Return true if count is odd, false otherwise
        return (count % 2 == 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (int(_x), int(_y))


def draw_zone_on_frame(frame, zone):
    coord = np.array(zone.coord, dtype=int)
    pts = coord.reshape((-1, 1, 2))
    color = compute_color_for_labels(randint(0, 100))

    frame = cv2.polylines(frame, [pts], True, color, 2)
    org = centroid(coord)
    frame = cv2.putText(frame, zone.zoneID, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 3)
    return frame