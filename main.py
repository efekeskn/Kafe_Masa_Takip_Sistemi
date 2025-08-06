import cv2
import numpy as np 
from ultralytics import YOLO

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def masa_kontrol(video_path, table_coordinates):
    model = YOLO("yolo11x.pt")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video acilamadi: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    table_polygon = np.array(table_coordinates, np.int32)
    
    kisiler = {}
    sureler = {}
    frame_sayisi = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_sayisi += 1
        zaman = frame_sayisi / fps
        
        results = model.track(frame, persist=True, classes=[0], conf=0.1, iou=0.5)
        
        cv2.polylines(frame, [table_polygon], isClosed=True, color=(0, 255, 0), thickness=3)
        y_min_polygon = np.min(table_polygon[:, 1])
        cv2.putText(frame, "MASA", (table_polygon[0][0], y_min_polygon - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
        
        masada = 0
        aktif_idler = set()
        
        if results[0].boxes is not None:
            tracking_var_mi = results[0].boxes.id is not None
            
            if tracking_var_mi:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                guvenler = results[0].boxes.conf.cpu().numpy()
                
                for box, id, guven in zip(boxes, ids, guvenler):
                    x_center, y_center, w, h = box
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    merkez_x = int(x_center)
                    merkez_y = int(y_center)
                    
                    merkez_masada = point_in_polygon((merkez_x, merkez_y), table_polygon)
                    
                    masada_mi = merkez_masada
                    
                    if masada_mi:
                        masada += 1
                        aktif_idler.add(id)
                        
                        if id not in kisiler:
                            kisiler[id] = {
                                "baslama": zaman,
                                "son_gorulen": frame_sayisi,
                            }
                        else:
                            kisiler[id]["son_gorulen"] = frame_sayisi
                        
                        sure = zaman - kisiler[id]["baslama"]
                        sureler[id] = sure
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(frame, f"ID:{id} - {sure:.1f}s", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                boxes = results[0].boxes.xywh.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x_center, y_center, w, h = box
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    merkez_x = int(x_center)
                    merkez_y = int(y_center)
                    
                    if point_in_polygon((merkez_x, merkez_y), table_polygon):
                        masada += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(frame, f"Kisi {i+1}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        silinecekler = []
        for id in kisiler:
            if id not in aktif_idler:
                kac_frame_gecti = frame_sayisi - kisiler[id]["son_gorulen"]
                if kac_frame_gecti > fps * 2:
                    silinecekler.append(id)
        
        for id in silinecekler:
            kisiler.pop(id, None)
            sureler.pop(id, None)
        
        panel_yukseklik = 80 + len(sureler) * 25
        cv2.rectangle(frame, (10, 10), (400, panel_yukseklik), (0, 0, 0), -1)
        
        cv2.putText(frame, f"MASADAKI KISI: {masada}", 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if sureler:
            cv2.putText(frame, "MASADA OTURANLAR:", 
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            y = 90
            for id, sure in sureler.items():
                cv2.putText(frame, f"ID {id}: {sure:.1f}s", 
                           (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y += 25
        else:
            cv2.putText(frame, "Kimse yok", 
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Masa Kontrolu", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Video suresi: {frame_sayisi / fps:.1f} saniye")
    if sureler:
        print("Masada oturanlar:")
        for id, sure in sureler.items():
            print(f"  ID {id}: {sure:.1f} saniye oturdu")
    else:
        print("Masada kimse yok.")

if __name__ == "__main__":
    table_coordinates =  [[1443, 493], [1655, 648], [1756, 476], [1580, 324], [1458, 448]]
    video_path = "video.mp4"
    
    masa_kontrol(video_path, table_coordinates)
    
