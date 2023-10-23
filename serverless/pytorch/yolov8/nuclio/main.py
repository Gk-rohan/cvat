import base64
from ultralytics import YOLO
import json
import io
from PIL import Image
import yaml
import numpy as np
from skimage.measure import find_contours, approximate_polygon
import cv2

def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read the DL model
    model = YOLO("best_paint.pt")
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolo-v8 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.1))
    context.user_data.model.conf = threshold
    image = Image.open(buf)
    yolo_results = context.user_data.model.predict(source=image, stream=True)

    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # print(yolo_results)
    encoded_results = []
    for result in yolo_results:
        # print(result.boxes.cls)
        # print(result.masks.xy)
        # context.logger.info(result.masks.xy)
        for idx, cls in enumerate(result.boxes.cls):
            mask = result.masks.data[idx].numpy()
            mask = mask.astype(np.uint8)
            mask = cv2.resize(mask,
                dsize=(image.width, image.height),
                interpolation=cv2.INTER_NEAREST)

            contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour = np.flip(contour, axis=1)
                if len(contour) < 3:
                    continue

                x_min = max(0, int(np.min(contour[:,:,0])))
                x_max = max(0, int(np.max(contour[:,:,0])))
                y_min = max(0, int(np.min(contour[:,:,1])))
                y_max = max(0, int(np.max(contour[:,:,1])))

                cvat_mask = to_cvat_mask((x_min, y_min, x_max, y_max), mask)
            # contours = find_contours(mask, 0.1)
            # contour = contours[0]
            # contour = np.flip(contour, axis=1)
            #     # Approximate the contour and reduce the number of points
            # contour = approximate_polygon(contour, tolerance=2.5)
            # if len(contour) < 6:
            #     continue

            # Xmin = int(np.min(contour[:,0]))
            # Xmax = int(np.max(contour[:,0]))
            # Ymin = int(np.min(contour[:,1]))
            # Ymax = int(np.max(contour[:,1]))
            # cvat_mask = to_cvat_mask((Xmin, Ymin, Xmax, Ymax), mask)
                # context.logger.info(contour.ravel())
                encoded_results.append({
                    'confidence': result.boxes.conf[idx].item(),
                    'label': labels[cls.item()],
                    'points': contour.ravel().tolist(),
                    'mask': cvat_mask,
                    'type': 'mask'
                })

    # context.logger.info(encoded_results)

    return context.Response(body=json.dumps(encoded_results), headers={},
                            content_type='application/json', status_code=200)