import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

def cafe_table_occupancy_detection(video_path, table_coordinates):
    """
    Cafe görüntülerinde masa doluluk tespiti yapan ana fonksiyon
    
    Args:
        video_path (str): Video dosyasının yolu
        table_coordinates (list): Masa koordinatları [x_min, y_min, x_max, y_max]
    """
    
    print("YOLO11 modeli yükleniyor...")
    model = YOLO('yolo11n.pt')  
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı - {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video özellikleri: {width}x{height}, {fps} FPS")
    
    x_min, y_min, x_max, y_max = table_coordinates
    
    person_tracks = {}  
    next_person_id = 1
    person_durations = {}  
    frame_count = 0
    
    track_history = defaultdict(lambda: deque(maxlen=30))
    
    print("Video işleme başlıyor... (Çıkmak için 'q' tuşuna basın)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti veya frame okunamadı")
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        results = model.track(frame, persist=True, classes=[0])  # class 0 = person
        
        # Masa bölgesini çiz
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "Hedef Masa", (x_min, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        people_in_table = 0
        current_frame_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                if conf < 0.5:  
                    continue
                    
                x_center, y_center, w, h = box
                x1 = int(x_center - w/2)
                y1 = int(y_center - h/2)
                x2 = int(x_center + w/2)
                y2 = int(y_center + h/2)
                
                person_center_x = int(x_center)
                person_center_y = int(y_center)
                
                if (x_min <= person_center_x <= x_max and 
                    y_min <= person_center_y <= y_max):
                    
                    people_in_table += 1
                    current_frame_ids.add(track_id)
                    
                    if track_id not in person_tracks:
                        person_tracks[track_id] = {
                            'start_time': current_time,
                            'last_seen': frame_count
                        }
                        print(f"Yeni kişi masaya oturdu - ID: {track_id}")
                    else:
                        person_tracks[track_id]['last_seen'] = frame_count
                    
                    duration = current_time - person_tracks[track_id]['start_time']
                    person_durations[track_id] = duration
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    duration_text = f"ID:{track_id} - {duration:.1f}s"
                    cv2.putText(frame, duration_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Takip geçmişini güncelle
                    track_history[track_id].append((person_center_x, person_center_y))
                    
                    # Hareket izini çiz
                    points = np.array(track_history[track_id], dtype=np.int32)
                    if len(points) > 1:
                        cv2.polylines(frame, [points], False, (0, 0, 255), 2)
        
        # Masadan ayrılan kişileri temizle (5 saniye görmezse ayrıldı kabul et)
        ids_to_remove = []
        for track_id in person_tracks:
            if track_id not in current_frame_ids:
                frames_since_last_seen = frame_count - person_tracks[track_id]['last_seen']
                if frames_since_last_seen > fps * 3:  # 3 saniye
                    final_duration = person_durations.get(track_id, 0)
                    print(f"Kişi masadan ayrıldı - ID: {track_id}, Toplam süre: {final_duration:.1f}s")
                    ids_to_remove.append(track_id)
        
        # Ayrılan kişileri kaldır
        for track_id in ids_to_remove:
            if track_id in person_tracks:
                del person_tracks[track_id]
            if track_id in person_durations:
                del person_durations[track_id]
            if track_id in track_history:
                del track_history[track_id]
        
        # Bilgi paneli
        info_y = 30
        cv2.putText(frame, f"Masadaki Kisi Sayisi: {people_in_table}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Aktif kişilerin sürelerini göster
        info_y += 30
        for track_id, duration in person_durations.items():
            cv2.putText(frame, f"ID {track_id}: {duration:.1f}s", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += 25
        
        # Frame bilgisi
        cv2.putText(frame, f"Frame: {frame_count} | Zaman: {current_time:.1f}s", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Sonucu göster
        cv2.imshow('Cafe Masa Doluluk Tespiti', frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları temizle
    cap.release()
    cv2.destroyAllWindows()
    
    # Özet bilgileri yazdır
    print("\n=== ÖZET ===")
    print(f"Toplam işlenen frame: {frame_count}")
    print(f"Video süresi: {frame_count/fps:.1f} saniye")
    if person_durations:
        print("Son durumda masadaki kişiler:")
        for track_id, duration in person_durations.items():
            print(f"  ID {track_id}: {duration:.1f} saniye")
    else:
        print("Video sonunda masada kimse yok.")

# Örnek kullanım
if __name__ == "__main__":
    # Video dosyası yolu
    video_path = "cafe_video.mp4"  # Kendi video dosyanızın yolunu yazın
    
    # Masa koordinatları [x_min, y_min, x_max, y_max]
    # Bu koordinatları video üzerinde masanın bulunduğu bölgeye göre ayarlayın
    table_coordinates = [300, 200, 600, 500]  # Örnek koordinatlar
    
    print("=== CAFE MASA DOLULUK TESPİTİ ===")
    print("YOLO11 kullanılarak masa doluluk analizi")
    print(f"Video: {video_path}")
    print(f"Masa koordinatları: {table_coordinates}")
    print("\nKurulum talimatları:")
    print("1. pip install ultralytics opencv-python numpy")
    print("2. İlk çalıştırmada YOLO11 modeli otomatik indirilecek")
    print("3. Video dosyasının yolunu doğru şekilde belirtin")
    print("4. Masa koordinatlarını videonuza göre ayarlayın\n")
    
    # Ana fonksiyonu çalıştır
    cafe_table_occupancy_detection(video_path, table_coordinates)