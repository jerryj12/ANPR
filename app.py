import csv
import time
from flask import Flask, redirect, render_template, request, send_file, send_from_directory, make_response
import cv2
import numpy as np
import os
from flask_ngrok import run_with_ngrok
from ultralytics import YOLO
from util import read_license_plate, write_csv
from scipy.interpolate import interp1d
import ast
import pandas as pd

app = Flask(__name__,static_url_path='')
run_with_ngrok(app)
# Define the upload folder
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def record_video(video_path, duration=10):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    start_time = time.time()
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video(input_path):
    # Define variables used in process_video
    coco_model = YOLO('yolov8n.pt')
    np_model = YOLO('license_plate_detector.pt')
    # mot_tracker = sort.Sort()
    results = {}
    output_path = os.path.join('static', 'output_video.mp4')
    global input
    input = output_path
    video = cv2.VideoCapture(input)
    vehicles = [2]
    ret = True
    global frame_number  # Declare frame_number as global
    frame_number = -1  # Reset frame_number for each video processing
    

    while ret:
        frame_number += 1
        ret, frame = video.read()

        if ret:
            results[frame_number] = {}
            detections = coco_model.track(frame, persist=True)[0]
            try:
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, track_id, score, class_id = detection

                    if int(class_id) in vehicles and score > 0.5:
                        # vehicle_bounding_boxes = []
                        # vehicle_bounding_boxes.append([x1, y1, x2, y2, track_id, score])
                        
                        # for bbox in vehicle_bounding_boxes:
                            roi = frame[int(y1):int(y2), int(x1):int(x2)]
                            license_plates = np_model(roi)[0]

                        for license_plate in license_plates.boxes.data.tolist():
                            plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate
                            plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]

                            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                            _, plate_treshold = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                            # np_text, np_score = read_license_plate(plate_treshold)
                            np_text, np_score = read_license_plate(plate_gray)

                            if np_text is not None:
                                # np_text = 'Invalid'
                                # np_score = '0'

                                results[frame_number][track_id] = {
                                    'car': {
                                        'bbox': [x1, y1, x2, y2]
                                    },
                                    'license_plate': {
                                        'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                        'bbox_score': plate_score,
                                        'text': np_text,
                                        'text_score': np_score
                                    }
                                }

            except:
                pass

    write_csv(results, './results.csv')
    video.release()

    obj = Visualize()
    obj.visual()

    return output_path, None  # Return None for the CSV path
   
# Route to upload the video file
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    scroll_to_download = False  # Default value
    csv_null = False
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded file to the static/assets folder
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
            file.save(video_path)
            # Process the video
            processed_video_path, output_csv_path = process_video(video_path)
            file.save(processed_video_path)
            output_csv_path = 'test_interpolated.csv'
            with open(output_csv_path, 'r') as file:
                reader = csv.reader(file)
                data_length = len(list(reader))
            
    # Check if the length is less than 1
            is_csv_empty = data_length < 2

            if is_csv_empty:
                csv_null = True
            # Return download links
            scroll_to_download = True

            return render_template('index.html',
                                   video_path=processed_video_path,
                                   csv_path=output_csv_path,
                                   scroll_to_download=scroll_to_download,csv_null=csv_null)

    return render_template('index.html', scroll_to_download=scroll_to_download)


#Detector  visual

from scipy.interpolate import interp1d
import ast
import pandas as pd
import csv
import numpy as np


class Visualize:

  def interpolate_bounding_boxes(self,data):
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

            interpolated_data.append(row)

    return interpolated_data

   

  def visual(self):
      with open('results.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

        # Interpolate missing data
        interpolated_data = self.interpolate_bounding_boxes(data)
        #print(interpolated_data)    
        # Write updated data to a new CSV file
        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        with open('test_interpolated.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)



      results = pd.read_csv('test_interpolated.csv')

      # load video
      video_path = input
      cap = cv2.VideoCapture(video_path)
      if os.path.exists('output.webm'):
          os.remove('output.webm')
      fourcc = cv2.VideoWriter_fourcc(*'VP80')  # Specify the codec
      fps = cap.get(cv2.CAP_PROP_FPS)
      width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      out = cv2.VideoWriter(f'output.webm', fourcc, fps, (width, height))

      license_plate = {}
      for car_id in np.unique(results['car_id']):
          max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
          license_plate[car_id] = {'license_number': results[(results['car_id'] == car_id) &
                                                            (results['license_number_score'] == max_)]['license_number'].iloc[0]}
          cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                  (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
#           ret, frame = cap.read()

#           x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
#                                                     (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

#           license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
#           license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))


#           license_plate[car_id]['license_crop'] = license_crop


      frame_nmr = -1

      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

      # read frames
      ret = True
      while ret:
          ret, frame = cap.read()
          frame_nmr += 1
          if ret:
              df_ = results[results['frame_nmr'] == frame_nmr]
              for row_indx in range(len(df_)):
                  # draw car
                  car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                  # draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                              # line_length_x=200, line_length_y=200)
                  cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0),12)



                  # draw license plate

                  roi = frame[int(car_y1):int(car_y2), int(car_x1):int(car_x2)]

                  x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                  cv2.rectangle(roi, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)


                  # draw box and writing license number

                  ### using general formula
                  ### set your own axis to make white boxes to put license numbers of cars.

                  text = license_plate[df_.iloc[row_indx]['car_id']]['license_number']
                  (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                  # cy1 = np.abs(int(y1) - int((y2 - y1) * 0.5))
                  # cy2 = np.abs(int(y1) + int((y2 - y1) * 0.5))
                  cy1 = np.abs(int(y1) - int((y2 - y1)))
                  cy2 = np.abs(int(y1) - 10)
                  cx1 = np.abs(int(x1 - (x2 - x1) * 0.2))
                  cx2 = np.abs(int(x2 + (x2 - x1) * 0.2))

                  roi[cy1:cy2, cx1:cx2, :] = (255, 255, 255)
                  cv2.rectangle(roi, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)

                  font_size = min((cy2 - cy1) // 2, (cx2 - cx1) // len(text))

                  cv2.putText(roi, text, (int(cx1 + 10), int(cy1 + (cy2 - cy1) // 1.5)),
                  cv2.FONT_HERSHEY_DUPLEX, font_size / 30, (0, 0, 0), 1, cv2.LINE_AA)


              out.write(frame)
              frame = cv2.resize(frame, (1280, 720))



      out.release()

      cap.release()



# Route to download the processed video
@app.route('/download_video')
def download_video():
    filename = 'output.webm'

    # Create a response object
    response = make_response(send_from_directory('', filename, conditional=True))

    # Add cache headers to prevent caching
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response

# Route to download the CSV file
@app.route('/download_csv')
def download_csv():
    filename = 'test_interpolated.csv'
    return send_file(filename, as_attachment=True)

@app.route('/record', methods=['POST'])
def record():
    video_name = f"output_video.mp4"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)

    record_video(video_path)
    output_csv_path = 'test_interpolated.csv'
    process_video(video_path)
    return render_template('index.html',
                                   video_path=video_path,
                                   csv_path=output_csv_path)

if __name__ == '__main__':
    app.run()
#debug=True
