# 🧠 FaceVision AI - Yapay Zeka Yüz Tanıma & Model Eğitimi Web Platformu

Öğretmeniniz ve ders projeniz için özel olarak tasarlanmış, **görseli döndürme (rotation), küçültme/büyütme (scaling), aynalama (flip), ışık/kontrast değişimi** veya **farklı fotoğraflarda aynı kişileri tanıma** yeteneğine sahip 128-boyutlu vektör tabanlı yüz tanıma platformu.

---

## 🌟 Projenin Önemli Özellikleri

1. **Gelişmiş Yüz Tespiti & Etiketleme (Model Eğitimi):**
   - Yüklenen fotoğraftaki veya kameradaki yüzleri otomatik tespit eder.
   - Her yüz için **128-boyutlu biyometrik vektör (Face Descriptor Embedding)** üretir.
   - Arkadaşlarınızın yüzlerine isim vererek kişiselleştirilmiş AI modelinizi kolayca eğitebilirsiniz.

2. **Dönüştürme & Dayanıklılık Test Laboratuvarı (Data Augmentation & Invariance):**
   - Fotoğrafı **0° - 360° döndürme**, **%30 - %250 ölçekleme (küçültme/büyütme)**, **parlaklık/kontrast** ve **aynalama** işlemlerine tabi tutsanız dahi, yapay zeka kişiyi tanımaya devam eder.
   - Hizalanmış biyometrik nirengi noktaları (Landmarks) sayesinde geometrik deformasyonlara karşı tam koruma sağlar.

3. **Çoklu Fotoğrafta Otomatik Tanıma (Cross-Photo Recognition):**
   - Arkadaş grubunuzun yer aldığı yeni fotoğraflar yükleyin.
   - Sistem fotoğraftaki herkesi tarar ve **Kosinüs Benzerliği (Cosine Similarity)** ile veri tabanındaki kişileri %0 - %100 güven oranıyla eşleştirir.

4. **Canlı Kamera (Webcam) Taraması:**
   - Web kameranızı açarak gerçek zamanlı yüz takibi ve kişi tanıma yapabilirsiniz.

5. **JSON Formatında Model Veri Seti Dışa/İçe Aktarma:**
   - Eğittiğiniz tüm modelleri `.json` dosyası olarak bilgisayarınıza indirebilir, hocanıza sunum sırasında ödev dosyası olarak teslim edebilirsiniz.

6. **Hocaya Sunum Rehberi & Matematiksel Altyapı:**
   - Uygulama içinde yer alan **"6. Hocaya Sunum Rehberi"** sekmesinde evrişimli sinir ağları (CNN), 128D vektör matrisleri ve Kosinüs Benzerliği matematik formülleri hazır durumdadır.

---

## 🚀 Projeyi Çalıştırma Rehberi

1. `index.html` dosyasına çift tıklayarak herhangi bir modern web tarayıcısında (Google Chrome, Microsoft Edge, Mozilla Firefox vb.) doğrudan çalıştırabilirsiniz.
2. İnternet bağlantısı veya ekstra sunucu kurulumu gerektirmez, %100 taranabilir ve istemci tarafında (Client-Side HTML5 Canvas + JS Engine) yüksek performansla çalışır.

---

## 🎓 Hocaya Sunumda Anlatılacak 5 Temel Adım

1. **Yüz Tespiti (Detection):** Görüntü matrisinden yüz bölgesinin koordinatları (X, Y, W, H) çıkartılır.
2. **Hizalama (Alignment):** Gözler ve burun açısı referans alınarak yüz dik pozisyona getirilir (döndürme invaryansı sağlanır).
3. **Derin Öznitelik Çıkarımı (128D Embeddings):** Yüz yapısı 128 adet reel sayıdan oluşan benzersiz bir vektöre dönüştürülür:
   $$\mathbf{V} = [v_1, v_2, v_3, \dots, v_{128}]$$
4. **Kosinüs Benzerliği (Cosine Similarity):** Bilinmeyen yüz vektörü ile kayıtlı kişiler vektörel açı karşılaştırmasına tabi tutulur:
   $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
5. **Veri Artırımı (Augmentation):** Görüntünün piksel parlaklığı veya açısı değişse de 128D vektör oranı korunduğu için model yüksek başarı oranına ulaşır.

---

### 📁 Proje Dosya Yapısı
- `index.html`: Kullanıcı arayüzü, sekmeler ve sunum panelleri.
- `styles.css`: Glassmorphism koyu tema CSS tasarım sistemi.
- `app.js`: Yüz tespiti, 128D vektör üretimi, veri artırımı dönüşüm motoru ve JSON veri tabanı mantığı.
- `assets/hero_banner.png`: Görsel biyometrik arayüz afişi.
