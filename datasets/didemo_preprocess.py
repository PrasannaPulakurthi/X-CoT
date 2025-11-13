
import csv
import cv2
import time

from datasets.video_capture import VideoCapture


def video_to_tensor_xpool_simplified(video_path):
    
    imgs, idxs = VideoCapture.load_frames_from_video(video_path, 12)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        time.sleep(1)
        cap = cv2.VideoCapture(video_path)
        print('cannot open', video_path)

    return cap.isOpened()

def preprocess_didemo(
        input_csv_dir,
        output_csv_dir):

    video_clip_pth = 'data/Didemo/test_videos/'


    # rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)

    for subset in ['test', 'train']:
        print('subset: {}'.format(subset))

        # load
        input_csv_pth = input_csv_dir + f'didemo_{subset}_label.csv'
        output_csv_pth = output_csv_dir + f'didemo_{subset}_label_clean.csv'
        print(f'input_csv_pth={input_csv_pth}')
        print(f'output_csv_pth={output_csv_pth}')

        collect_lines = []
        invalid_video_ls = []

        # read csv file
        with open(input_csv_pth, "r", newline="") as csv_file:
            csv_reader = csv.reader(csv_file)

            for i, row in enumerate(csv_reader):
                print(f'i={i}, row={row}, row type={type(row)}')
                # i=3448, row=['True',
                # '/home/ubuntu/search_efs/public_datasets/didemo/all_videos/34025889@N00_5269925856_b09f28f7c0.mp4',
                # 'tensor([[0]])',
                # 'tensor([[31]])',
                # 'a woman squirts the whipped cream out of the can. woman starts spraying whipped cream the person is putting whipped cream on the coffee.',
                # '/home/ubuntu/search_efs/public_datasets/didemo/formatted_data/train_data_id_3461.pth'],
                # row type=<class 'list'>

                row_content = row
                print(row[1].split('/'), '\n')
                # ['', 'home', 'ubuntu', 'search_efs', 'public_datasets', 'didemo', 'all_videos', '18587146@N00_7890568006_2b64a33ed7.wmv']
                video_name = row[1].split('/')[-1]

                video_pth = video_clip_pth + video_name
                video_state = video_to_tensor_xpool_simplified(video_pth)

                if video_state:
                    print(f'can open video, keep')
                    collect_lines.append(row_content)
                else:
                    print(f'failed to open video, drop')
                    invalid_video_ls.append(video_name)

            print(f'\nFor subset={subset} finish processing all videos!\n')

        # write to csv
        with open(output_csv_pth, "w", newline="") as output_csv_file:
            csv_writer = csv.writer(output_csv_file)

            for row in collect_lines:
                csv_writer.writerow(row)

        print(f'Finish!')
        print(f'in subset={subset}, there are in total {len(invalid_video_ls)} are invalid, they are dropped')


input_csv_dir = 'data/Didemo/'
output_csv_dir = 'data/Didemo/'
preprocess_didemo(input_csv_dir, output_csv_dir)

