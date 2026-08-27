# DRONIONS — Ölçülmüş Bulgular

Bu dosya, simülasyonda **ölçülmüş** sonuçların kaydıdır. Her madde bir sayıya
dayanır ve o sayının nereden geldiğini söyler. Tahminler, hipotezler ve
"muhtemelen" cümleleri buraya girmez — çürütülmüş hipotezler ise ayrı bir
başlıkta durur, çünkü neyin denenip tutmadığı da bir sonuçtur.

**Kapsam uyarısı, her sayı için geçerli:** bütün ölçümler tek bir odada, tek
ışık altında, Gazebo render'ıyla yapılmıştır. Simülasyonda neyin ne kadar
kazandırdığını söyler; gerçek bir oda hakkında bir şey söylemez.

---

## 1. Dedektör: açık sözlük ile göreve özel modelin karşılaştırması

Aramanın gerçekten uçtuğu bakış noktalarında, aynı karelerde, aynı puanlama
kuralıyla (`scripts/compare_detectors.py`, `logs/detector_comparison.csv`).
Bir tespit ancak nesnenin gerçekte bulunduğu yeri kapsıyorsa ve geometrinin
öngördüğü genişliğe yakınsa sayılır.

| nesne | görünür | YOLO-World | fine-tune edilmiş YOLOv8n |
|---|---|---|---|
| laptop | 8 | 6 | 8 |
| book | 8 | 2 | 8 |
| mug | 8 | 6 | 8 |
| phone | 8 | 1 | 8 |
| box | 4 | 4 | 4 |
| **toplam** | **36** | **19 (%53)** | **36 (%100)** |

**Anlamı:** kapılar, VLM ve takip yalnızca dedektörün *zaten bulduğunu* görür.
Telefonu sekiz bakıştan birinde bulan bir dedektörün arkasındaki hiçbir
iyileştirme o sayıyı kıpırdatamaz. Haftalarca kapı ayarlamanın darboğazı buydu.

**Güven değerleri kıyaslanabilir değil.** Doğru tespitlerde açık modelin
ortancası 0.64, eğitilmiş modelinki 0.37. Eğitilmiş model daha az emin ama daha
sık haklı — küçük nesnede dar kutu çizmek emin olunması zor bir iş. Ham hâlde
birleştirilirse açık model her sıralamayı haklılıkla ilgisiz bir sebeple
kazanır. `perception/hybrid.py` bu yüzden her iki güveni kendi ortancasına
bölerek sıralar.

**Takas, "hangisi daha iyi" değil.** Eğitilmiş model yalnızca gösterildiği beş
nesneyi bilir; projenin önermesi ise kullanıcının kendi kelimeleriyle
konuşması. Hibrit ikisini birlikte çalıştırır: açık model her şeye cevap verir,
eğitilmiş model kendi sınıflarında yetkilidir.

---

## 2. Kamera ışını gövde merkezinden değil lensten atılmalı

Kamera `base_link`'e göre 0.12 m ileri ve 0.242 m yukarıda monteli
(`px4/models/x500_dronions/model.sdf`). `spatial.py` içindeki her projeksiyon
ışını `base_link`'ten atıyordu.

Hata bir kaldıraç kolu, yani nesne yaklaştıkça büyür. Gerçek yöne göre ölçülen
sapma, 24 tespit:

| | ortanca | 0.45 m'de |
|---|---|---|
| gövde merkezinden | 15.0° | 43° |
| lensten | **2.9°** | **7.7°** |

Konum hatası 26 tespitte **0.24 m → 0.15 m** (ortanca), 26'nın 18'i düzeldi.

**Yan sonuç:** düzlem projeksiyonu her bakış noktasında gerçek menzilin
0.61–0.81 katını veriyordu. Bu "alçalırken geometri bozuluyor" diye okunmuş ve
`RANGE_TRUST_RATIO` bu yüzden konmuştu. Geometri sağlammış, yanlış noktadan
atılıyormuş.

---

## 3. Aracın sapma kestirimi kendi yönü değil

MAVROS'un bildirdiği sapma gerçek yönelimden farklı, konumu ise doğru
(`scripts/measure_heading_bias.py`). Işın doğru noktadan çıkıp yanlış yöne
gider; hata yanal ve mesafeyle büyür.

- Beş açılışta ölçülen önyargı: **−4.16, −4.18, −4.99, −6.13, −5.98 derece**.
  Her biri yeni bir EKF hizalanması, yani koda gömülemez.
- Bir koşu içinde: kalkıştan hemen önce −5.98, sekiz dakika sonra −3.90,
  sonraki iki dakikada yalnızca 0.12°. Kayma kalkış civarında olur, sonra
  yavaşlar.
- Doğru ölçüm anı bu yüzden **uçuştan hemen önce**; hedef kalkıştan ~1 dakika
  sonra konumlandırılıyor.

Aynı nesne, aynı başlangıç noktası, dört koşu:

| koşu | tahmin | hata |
|---|---|---|
| düzeltmesiz | (3.00, −0.58) | 0.363 m |
| düzeltmesiz | (2.89, −0.63) | 0.440 m |
| sabit −4.18° | (2.99, −0.38) | 0.171 m |
| **koşuda ölçülen −5.98°** | **(3.08, −0.30)** | **0.085 m** |

Gerçek kitap (3.05, −0.22).

**Rapora girecek dürüst çift:** algı hattının kendi doğruluğu **~0.09 m**,
aracın yön kestirimi dahil doğruluk **~0.40 m**. Aradaki farkın gerçek
donanımdaki adı manyetometre kalibrasyonudur. Ölçüm düzeneği bunu yer
gerçeğinden okur, yani sistemin parçası değil deney kontrolüdür.

---

## 4. Menzil politikası: düzlem, boyut, ve melez

35 geçerli tespit, düzeltilmiş ölçüm düzeneğiyle.

| politika | ortanca | ortalama | en kötü | >0.5 m | \|dz\| |
|---|---|---|---|---|---|
| her zaman düzlem | 0.106 m | 0.275 m | 3.766 m | 4 | 0.06 m |
| her zaman boyut | 0.145 m | 0.167 m | 0.633 m | 1 | 0.12 m |
| **melez (mevcut, eşik 2.0)** | **0.106 m** | **0.155 m** | 0.676 m | **1** | **0.06 m** |
| melez, eşik 1.5 | 0.123 m | 0.158 m | 0.676 m | 1 | 0.06 m |
| melez, eşik 3.0 | 0.106 m | 0.267 m | 3.766 m | 3 | 0.06 m |

**Sonuç: değişiklik gerekmiyor.** Mevcut kural düzlemin ortancasını ve
yüksekliğini, boyutun aykırı direncini alıyor. Düzlem tek başına bazen felaket
ediyor (4.5 m), boyut daha yumuşak bozuluyor ama yüksekliği iki kat kötü.

---

## 5. Renk kapısı: yalnızca nesnenin kendi rengi referans olduğunda çalışır

16 tespit üzerinde, hedefin kendi nesnesi ve karıştırıcılar ayrı sayılarak.

| hedef | kendi nesnesi | karıştırıcılar | durum |
|---|---|---|---|
| book | 4/4 | **2/12** | çalışıyor |
| laptop | 4/4 | 12/12 | yapısal olarak ölü |
| mug | 3/4 | 8/12 | zayıf, kendi nesnesini de reddediyor |
| phone | 4/4 | 7/12 | zayıf, ama kitabı 4/4 eliyor |

**Mekanizma:** kapı referanslar üzerinde VEYA işletir. Ahşap masayı yakalayan
**tek bir referans** bütün seti o masadaki her şeye açar. Kitabın eski dört
referansından ikisi ahşaptı (ton 13 ve 26); yeniden üretilince set baştan sona
mavi kapak oldu (220–223) ve kapı 12/12'den 2/12'ye indi.

**Laptop yapısal olarak çözülemez:** ekranının mavisi kitabın kapağının
mavisidir. Renk bu ikisini ayıramaz, referans ne kadar temiz olursa olsun.

**Denenip terk edilen:** segmentasyon maskesiyle kırpıp referansı nesnenin
kendi piksellerine indirgemek. Ölçüldü, daha kötü. Siyah telefonun kendi doymuş
pikselleri ton 37, masa 28 — telefon drondan bakınca zaten ahşap tonunda
görünür, ve kapı iki maskesiz drone görüntüsünü karşılaştırır. Maskeleyince
telefon kapısı kitabı elemeyi bırakıyor (4/4 → 0/4), kitabınki 2/12'den
12/12'ye düşüyor.

---

## 6. Uçtan uca tekrarlanabilirlik

Dört hedef, ikişer gerçek uçuş, hepsi orijinden.

- **8/8** koşuda devretme oldu ve hedefe varıldı.
- **7/8** doğru nesneye. Tek sapma telefon koşusuydu; tahmin 0.73 m kaydı ve
  şişeye telefondan yakın düştü.
- Sıfır çarpma, sıfır erken varış reddi, sıfır kota olayı.
- Hedef verildikten varışa süre: laptop 23 s, kupa 18–23 s, kitap 26 s.

**Ölçümün sınırı:** koşuların hepsi aynı noktadan yaklaştı, çünkü eğitilmiş
dedektör hedefi kalkış noktasından buluyor ve arama neredeyse hiç çalışmıyor.
Yani bu sayılar "bulma ve yaklaşma çalışıyor" der, "her geometride çalışıyor"
demez.

---

## 7. Konum çözünürlüğü ile nesne aralığı

Telefon (2.62, −0.16) ile kitap (3.05, −0.22) arası **0.44 m**. Ölçülen konum
hatası 0.11–0.23 m. Yani hata aralığın yarısına yaklaşıyor ve **etiketi
çevirebiliyor**: bir koşuda tahmin (2.85, −0.18) geldi — telefona 0.23 m,
kitaba 0.20 m. Drone kitaba gitmedi, ama puanlama onu kitap saydı.

Bu, "hangi nesneye gittiğini ne kadar güvenle söyleyebiliriz" sorusunun sayısal
cevabıdır ve bu boru hattının şu anki sınırıdır.

---

## 8. Gemini ücretsiz katman kotası ölçümü bozuyor

Sınır **günde 20 çağrı**, proje ve model başına. Kota dolduğunda SDK 429'u
kendi içinde üstel geri çekilmeyle yeniden dener ve ancak başardığında döner.

**Ölçülen:** günün 19. çağrısında tek bir arama çağrısı **4 dakika 46 saniye**
sürdü ve drone bunun tamamında kıpırdamadan asılı kaldı — hedefi ilk saniyede
tespit etmiş olmasına rağmen. Logda hiçbir şey bunu söylemiyordu.

Aynı uçuş, taze kotayla: çağrı **11.2 saniye**, toplam süre **26 saniye**.

**Sonuç:** kota tükenmişken alınan hiçbir zaman ölçümü geçerli değildir. Çağrı
artık 15 s zaman aşımı ve tek yeniden denemeyle sınırlı; 429 loglanıyor; her
çağrının süresi kaydediliyor.

---

## 9. Açık sözlük neyi tutuyor: mobilyayı, küçük nesneleri değil

YOLO-World'ün, eğitim verimizde **hiç bulunmayan** sekiz nesnedeki geri
çağırması (`scripts/measure_open_vocab.py`, `logs/open_vocab.csv`). 96 örneğin
94'ü geçerli.

| görülmemiş | | | görülmemiş | |
|---|---|---|---|---|
| chair | 5/5 | | bowl | **0/8** |
| table | 8/8 | | headphones | **0/5** |
| bookshelf | 8/8 | | bottle | **0/4** |
| sofa | 4/5 | | | |
| cabinet | 5/8 | | **toplam** | **30/51 (%59)** |

Bölünme keskin: **mobilyayı neredeyse kusursuz buluyor, küçük nesneleri hiç
bulamıyor.**

**Fine-tune kararı açısından anlamı:** açık sözlüğün burada koruduğu şey
mobilyadır, ve mobilya yüzey taramasının dayandığı şeydir — hedefin masada mı
koltukta mı olduğunu anlama yeteneği oradan gelir. YOLO-World'ü fine-tune etmek
bunu kaybettirirse, yalnızca "şarj aletimi bul" özelliği değil, **aramanın
yüzey mantığı** da gider. Kabul kriteri bunu da kapsamalıdır.

**Ölçüm notu:** aynı betik eğitilen dört nesnede %25 verir (8/32), oysa
`compare_detectors` %53 verir (19/36). İkisi aynı şeyi ölçmez: uçan sistem
küratörlü prompt listesi ve negatif prompt'lar kullanır, bu betik ise çıplak
sınıf adı — çünkü görülmemiş sekiz nesne için küratörlü liste yoktur ve iki
grubun aynı kuralla ölçülmesi gerekir. Bu sayı uçan sistemin performansı
değildir; fine-tune öncesi/sonrası karşılaştırması için tutarlı bir taban
çizgisidir.

---

## 10. YOLO-World'ü fine-tune etmek açık sözlüğü tamamen yok ediyor

Danışman önerisi: ayrı bir kapalı-kümeli model yerine YOLO-World'ün kendisini
oda verimizle fine-tune etmek. Önerilen dört önlem de uygulandı — `freeze=10`
(omurga donuk), `lr0=0.001`, 40 epoch sınırı, `patience=10` erken durdurma.
Eğitim 22 epoch'ta durdu, 20 dakika, zirve GPU 3.39 GiB, 960 piksel.

| | fine-tune öncesi | fine-tune sonrası |
|---|---|---|
| **görülmemiş sekiz nesne** | 30/51 (%59) | **0/42 (%0)** |
| eğitilen dört nesne | 8/32 (%25) | 28/28 (%100) |

Açık sözlük zayıflamadı, **yok oldu**. 42 örnekte tek tespit yok. Referansta
8/8 olan `table`, 8/8 olan `bookshelf`, 5/5 olan `chair` — hepsi sıfır.

**Ölçüm sağlam:** aynı `set_classes` yolu eğitilen dört sınıfta 28/28 veriyor,
yani mekanizma çalışıyor; model yalnızca görmediği kelimelere cevap vermiyor.

**Neden önlemler yetmedi:** YOLO-World'ün açık sözlüğü omurgada durmuyor.
Metin gömülmeleri (`txt_feats`, 1×80×512) ile görüntünün hizalanması modül 21
(`C2fAttn`) ve 22 (`WorldDetect`) içinde — `freeze=10`'un eğitilebilir bıraktığı
tam o iki modül. Omurgayı dondurmak genel görsel özellikleri korur, sözlüğü
korumaz.

**Kazanç tarafı hiçbir şey eklemiyor:** fine-tune edilmiş model kendi dört
nesnesinde %100 yapıyor, ama ayrı eğitilen YOLOv8n zaten 36/36 yapıyor
(bkz. bölüm 1). Yani takasta kazanılan tek şey "iki model yerine tek model",
kaybedilen şey projenin üzerine kurulduğu özellik ve aramanın yüzey mantığı.

**Karar: hibrit kalıyor.** Yukarı akıştaki kayıt da aynı yöne işaret ediyordu
(ultralytics#10038): orada da fine-tune edilmiş YOLO-World hem sıfır-atışı
kaybetmiş hem de kendi sınıflarında düz YOLOv8'in fine-tune halinden kötü
çıkmıştı.

---

## Çürütülmüş hipotezler

Neyin denenip tutmadığı da sonuçtur.

**"Konum hatası bakış açısından geliyor."** İki bağımsız ölçüm çürüttü. Gerçek
modelle beş koşu: 3°→0.23 m, 7°→0.11 m, 55°→0.12 m, 81°→0.04 m, 89°→0.10 m.
En büyük hata neredeyse cepheden bakan koşuda. Açı ile hata arasında ilişki
yok.

**"1280 piksel çıkarım küçük nesnelerde daha iyi bulur."** Yakın mesafe geri
çağırmasını yarıya düşürdü. YOLO-World 640'ta eğitilmiş ve oradan
oynatılamıyor.

**"YOLO-World'ün daha büyük veri seti bizim nesnelerimizde kesinliği
artırır."** Ölçüm desteklemiyor: ön eğitim bizim nesnelerimizde işe yaramıyor
(telefon 1/8, kitap 2/8). Darboğaz veri hacmi değil, bu render'da nesnelerin
20–40 piksel olması. Ayrıca yukarı akıştaki kayıtlı deneyde
(ultralytics#10038) fine-tune edilmiş YOLO-World, düz YOLOv8'in fine-tune
halinden **daha kötü** çıkmış ve sıfır-atış yeteneğini de kaybetmiş.

---

## Ölçüm düzeneğinde bulunan hatalar

Bu projede tekrar eden kalıp: ölçüm düzeneği ilk kurulduğunda sessizce yanlıştı
ve sonucu bir sistem hatası gibi okunabilirdi. Kaydı, çünkü kalıp devam ediyor.

| hata | nasıl görünüyordu | gerçekte ne olduğu |
|---|---|---|
| Yüzey taramasının girinti hatası | drone başlangıç noktasında "varıldı" diyor | dedektör mobilya modunda kalıyordu; masayı kutuluyordu |
| `place()` bakış noktasının altına iniyordu | konumlandırmada aykırı değerler | mobilyanın içine düşüp yana fırlıyordu; ölçülen bakış noktası istenen değildi |
| Doğuş noktası karton kutunun içinde | "uçuş anomalisi", çarpma sanıldı | gerçek temas fırlatması; açıklık kontrolü sadece mobilyaya bakıyordu |
| PX4 yerel orijini = doğuş noktası | yeni açılardan hedef bulunamıyor | dünya koordinatları kayıyordu; drone başka bölgeyi tarıyordu |
| PX4 statustext kanalı ölü | failsafe sebebi bilinmiyor | node abone ama tüm koşularda sıfır mesaj; sebep yayınlanıyor, kimse duymuyor |
| Anomali logu yalnızca mesafe yazıyordu | "3.0 m / 2.97 s" | gerçek uçuş ile kestirim sıçraması aynı görünüyordu; uçlar eklenince z = −4.5 m çıktı |

---

## Açık kalanlar

- Başlangıç konumu çeşitliliği: aracı taşımanın iki yolu da PX4'e çarptı
  (doğuş kaydırma → EKF ıraksaması; aktarım uçuşu → failsafe RTL). Üçüncü yol
  (tanımlamayı geciktirmek) uygulandı, ölçümü sürüyor.
- PX4 statustext kanalının neden sessiz olduğu.
- Telefonun 0.73 m'lik tek sapması — bakış açısı olmadığı kesinleşti, sebebi
  açık.
- Varış sonrası konum akışı düşmeleri — iki koşuda dörder kez, zararı görünmüyor.
- Fine-tune deneyi (hocanın önerisi): ölçüm betiği hazır, koşulmadı.
