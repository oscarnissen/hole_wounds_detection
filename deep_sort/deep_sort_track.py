######## DeepSORT -> Importing DeepSORT
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


class TrackDeepSort:
    def __init__(self, deepsort_model):
        # deepsort_model="C:/Users/ayas/OneDrive - SINTEF/Projects&FishMachineInteraction/src/Fish-detection' \
        #  '/deep_sort/resources/networks/mars-small128.pb"):
        ########################################
        # DeepSORT -> Initializing tracker.
        max_cosine_distance = 0.4
        nn_budget = None
        self.deepsort_model = deepsort_model
        self.encoder = gdet.create_box_encoder(self.deepsort_model, batch_size=1)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(self.metric)
        ########################################

    def track_fish(self, im0, bboxes, scores):
        # DeepSORT -> Getting appearance features of the object.
        featuresL = self.encoder(im0, bboxes)
        # DeepSORT -> Storing all the required info in a list.
        detectionsL = [Detection(bbox, score, feature) for bbox, score, feature in
                       zip(bboxes, scores, featuresL)]

        # DeepSORT -> Predicting Tracks.
        self.tracker.predict()
        self.tracker.update(detectionsL)
        # track_time = time.time() - prev_time
        tracker_ids = []
        cnt = 0

        # DeepSORT -> Plotting the tracks.
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # DeepSORT -> Changing track bbox to top left, bottom right coordinates.
            bbox = list(track.to_tlbr())
            # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
            # txt = 'id:' + str(track.track_id)
            #print("cnt: ", cnt)
            #print("bboxes: ", bboxes)
            tracker_ids.append([
                str(track.track_id),
                #int(bboxes[cnt][0]), int(bboxes[cnt][1]),
                #int(bboxes[cnt][2]), int(bboxes[cnt][3]),
                int(bbox[0]), int(bbox[1]), bbox[2], bbox[3], bbox[4],
                int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
            ])
            cnt += 1

        return tracker_ids


    def stereo_track_fish(self, im0, bboxes, scores):
        # DeepSORT -> Getting appearance features of the object.
        #print("  DeepSORT stereo_track_fish - bboxes: ", bboxes)
        features = self.encoder(im0, bboxes)
        # DeepSORT -> Storing all the required info in a list.
        detections = [Detection(bbox, score, feature) for bbox, score, feature in
                       zip(bboxes, scores, features)]

        # DeepSORT -> Predicting Tracks.
        self.tracker.predict()
        self.tracker.update(detections)
        # track_time = time.time() - prev_time
        tracker_ids = []
        cnt = 0

        # DeepSORT -> Plotting the tracks.
        for track in self.tracker.tracks:

            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # DeepSORT -> Changing track bbox to top left, bottom right coordinates.
            #print("DeepSORT tracking - before to_tlbr(): track.features: ", track.features)
            #print("DeepSORT tracking - before to_tlbr(): track.mean: ", track.mean)

            bbox = list(track.to_tlbr())
            # DeepSORT -> Writing Track bounding box and ID on the frame using OpenCV.
            # txt = 'id:' + str(track.track_id)
            #print("cnt: ", cnt)
            #print("Found ", len(bbox), "bboxes in DeepSORT tracking - after to_tlbr(): ")
            #for i, b in enumerate(bbox):
            #    print("box ", i, ": ", b)
            tracker_ids.append([
                int(track.track_id),
                #float(track.tlwh[0]), float(track.tlwh[1]),
                #float(track.tlwh[2]), float(track.tlwh[3]),
                #float(track.tlwh[4]),
                #track.confidence #,
                float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]),
                track.confidence
            ])
            cnt += 1

        return tracker_ids
