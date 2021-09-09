from collections import deque
import numpy as np
import cv2
import json

class Zones:
    def __init__(self, zones_polygon):
        with open(zones_polygon) as f:
            zone_data = json.load(f)

        zonelist = []
        for i in zone_data['shapes']:
            zone_id = i['label']
            coord = i['points']
            zone = Zone(coord, zone_id)
            zonelist.append(zone)
        self.zonelist = zonelist
        self.zonelist_buffer = deque(maxlen=int(40))
        self.frame_counter = 0
        self.frame_counter_reset = 400
        self.zones_tz_rec = np.zeros(len(self.zonelist), dtype=bool)
        self.zones_timer = 0

    def check_zone(self, point):
        '''
        check to which zone is this point inside
        Parameters
        ----------
        points: (x,y) point of the bottom middle bounding box representing the feet of person.

        Returns: (zoneindex) that the point belongs to
        -------

        '''
        xmid_feet, ybot_feet = point
        zone_listcheck = list(map(lambda z: z.is_inside_polygon((xmid_feet, ybot_feet)), self.zonelist))
        _tmp_out = [i for i, x in enumerate(zone_listcheck) if x]
        if len(_tmp_out) == 0:
            return None
        else:
            return _tmp_out[0]


    def update_frame_count(self):
        if self.frame_counter >= self.frame_counter_reset:
            self.zones_timer = 0
            self.zones_tz_rec = np.zeros(len(self.zonelist))
        else:
            self.zones_timer += 1


    def update_zone_still_count(self):
        nstill_zone = list(map(lambda x: x.n_still, self.zonelist))
        self.zonelist_buffer.append(nstill_zone)
        return nstill_zone


    def draw_zones(self, inframe):
        frame = np.copy(inframe)
        for i, zone in enumerate(self.zonelist):
            if self.zones_tz_rec[i] == False:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            frame = draw_zone_on_frame(frame, zone, color)
        return frame


class Zone(object):
    '''
    coord are stored as list of (x,y)
    '''
    def __init__(self, coord, zoneID):
        self.zoneID = zoneID
        self.coord = coord
        self.n_still = 0
        self.buffer_bnbs = []

    def _reset(self):
        self.n_still = 0
        self.buffer_bnbs = []

    def _add_still(self, bnb):
        self.n_still += 1
        self.buffer_bnbs.append(bnb)

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

    def get_zone_centroid(self):
        coord = np.array(self.coord, dtype=int)
        _x_list = [vertex[0] for vertex in coord]
        _y_list = [vertex[1] for vertex in coord]
        _len = len(coord)
        _x = sum(_x_list) / _len
        _y = sum(_y_list) / _len
        return (int(_x), int(_y))

    def get_bnb_pt_tz(self):
        if len(self.buffer_bnbs) == 1:
            bnb = self.buffer_bnbs[0]
            midx = bnb[0] + ((bnb[2] - bnb[0])/2)
            midy = bnb[1] + ((bnb[3] - bnb[1])/2)
            bnbh = bnb[3] - bnb[1]
            return (midx, midy, bnbh)
        else:
            min_bnby = np.inf
            max_bnby = -np.inf
            min_bnbx = np.inf
            max_bnbx = -np.inf
            for bnb in self.buffer_bnbs:
                bnbxleft, bnbytop, bnbxright, bnbybot = bnb
                if min_bnby > bnbytop:
                    min_bnby = bnbytop
                if max_bnby < bnbybot:
                    max_bnby = bnbybot
                if min_bnbx > bnbxleft:
                    min_bnbx = bnbxleft
                if max_bnbx < bnbxright:
                    max_bnbx = bnbxright

            midx = min_bnbx + ((max_bnbx - min_bnbx)/2)
            midy = min_bnby + ((max_bnby - min_bnby)/2)
            cbnbh = max_bnby - min_bnby

            return (midx, midy, cbnbh)


class ROIS(object):
    def __init__(self, rois_polygon):
        with open(rois_polygon) as f:
            roi_data = json.load(f)

        roislist = []
        for i in roi_data['shapes']:
            roi_id = i['label']
            coord = i['points']
            roi = ROI(coord, roi_id)
            roislist.append(roi)

        self.roislist = roislist
        self.roislist_buffer = deque(maxlen=int(40))


    def check_rois(self, point):
        '''
        check to which roi is this point inside
        Parameters
        ----------
        points: (x,y) point of the bottom middle bounding box representing the feet of person.

        Returns: (zoneindex) that the point belongs to
        -------

        '''
        xmid_feet, ybot_feet = point
        rois_listcheck = list(map(lambda z: z.is_inside_polygon((xmid_feet, ybot_feet)), self.roislist))
        _tmp_out = [i for i, x in enumerate(rois_listcheck) if x]
        if len(_tmp_out) == 0:
            return None
        else:
            return _tmp_out[0]


    def draw_roi(self, inframe):
        frame = np.copy(inframe)
        for roi in self.roislist:
            color = (0, 0, 255)
            frame = draw_rois_on_frame(frame, roi, color)
        return frame


class ROI(object):
    def __init__(self, coord, roiID):
        self.roiID = roiID
        self.coord = coord

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


def draw_zone_on_frame(frame, zone, color):
    coord = np.array(zone.coord, dtype=int)
    pts = coord.reshape((-1, 1, 2))
    color = color

    frame = cv2.polylines(frame, [pts], True, color, 2)
    org = centroid(coord)
    frame = cv2.putText(frame, zone.zoneID, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
    return frame


def draw_rois_on_frame(frame, roi, color):
    coord = np.array(roi.coord, dtype=int)
    pts = coord.reshape((-1, 1, 2))
    color = color

    frame = cv2.polylines(frame, [pts], True, color, 2)
    org = centroid(coord)
    frame = cv2.putText(frame, roi.roiID, org,
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
    return frame