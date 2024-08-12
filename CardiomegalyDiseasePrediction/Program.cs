using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

public class CardiomegalyPrediction
{
    private string trainDataPath;
    private string testDataPath;
    private int targetWidth;
    private int targetHeight;
    private int numClasses;
    private Sequential model;

    // Yapıcı metod: Eğitim ve test veri yolu ile hedef boyutları alır
    public CardiomegalyPrediction(string trainDataPath, string testDataPath, int targetWidth = 224, int targetHeight = 224)
    {
        this.trainDataPath = trainDataPath;
        this.testDataPath = testDataPath;
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
        this.numClasses = 2; // Çıktı sınıf sayısı (örneğin, kardiyomegali var/yok)
    }

    // Verilen dosya yolundan resmi yükler ve ön işler
    public static NDArray LoadImage(string filePath, int targetWidth, int targetHeight)
    {
        var img = tf.io.read_file(filePath); // Resim dosyasını oku
        var decoded = tf.image.decode_jpeg(img, channels: 3); // JPEG resmini çöz
        var resized = tf.image.resize(decoded, new[] { targetHeight, targetWidth }); // Boyutları yeniden ayarla
        resized = resized / 255.0f; // Piksel değerlerini [0, 1] aralığına normalize et
        return resized.numpy();
    }

    // Verilen dosya yolları ve etiketleri kullanarak veriyi yükler ve ön işler
    public static (NDArray images, NDArray labels) LoadData(List<string> filepaths, List<int> labels, int targetWidth, int targetHeight)
    {
        var imageArray = new List<NDArray>();
        foreach (var file in filepaths)
        {
            var image = LoadImage(file, targetWidth, targetHeight); // Her resmi yükle
            imageArray.Add(image);
        }

        var images = np.stack(imageArray.ToArray()); // Resimleri birleştir
        var labelsArray = np.array(labels.ToArray()); // Etiketleri numpy array'ine dönüştür

        return (images, labelsArray);
    }

    // Verilen veri yolundan dosya yollarını ve etiketleri alır
    private List<string> GetFilepaths(string dataPath, out List<int> labels)
    {
        var filepaths = new List<string>();
        labels = new List<int>();

        var classes = Directory.GetDirectories(dataPath); // Klasörleri al (her klasör bir sınıfı temsil eder)
        for (int i = 0; i < classes.Length; i++)
        {
            var files = Directory.GetFiles(classes[i]); // Her sınıf için dosyaları al
            filepaths.AddRange(files);
            labels.AddRange(new int[files.Length].Select(_ => i)); // Her dosya için etiket ekle
        }

        return filepaths;
    }

    // Modeli oluşturur ve derler
    public void BuildModel()
    {
        model = keras.Sequential(); // Modeli oluştur

        model.add(keras.layers.InputLayer(input_shape: (targetHeight, targetWidth, 3))); // Girdi katmanı
        model.add(keras.layers.Conv2D(32, kernel_size: (3, 3), activation: "relu", padding: "VALID")); // Konvolüsyon katmanı
        model.add(keras.layers.MaxPooling2D(pool_size: (2, 2), padding: "VALID")); // Maksimum havuzlama katmanı
        model.add(keras.layers.Conv2D(64, kernel_size: (3, 3), activation: "relu", padding: "VALID")); // İkinci konvolüsyon katmanı
        model.add(keras.layers.MaxPooling2D(pool_size: (2, 2), padding: "VALID")); // İkinci maksimum havuzlama katmanı
        model.add(keras.layers.Flatten()); // Yassılaştırma katmanı
        model.add(keras.layers.Dense(256, activation: "relu")); // Tam bağlantılı katman
        model.add(keras.layers.Dropout(0.5f)); // Dropout katmanı
        model.add(keras.layers.Dense(numClasses, activation: "softmax")); // Çıktı katmanı

        // Modeli derle
        model.compile(optimizer: keras.optimizers.Adam(learning_rate: 0.001f),
                      loss: keras.losses.SparseCategoricalCrossentropy(),
                      metrics: new[] { "accuracy" });
    }

    // Modeli eğitir ve değerlendirir
    public void TrainAndEvaluate(int epochs = 3) // Epoch sayısını belirler, varsayılan olarak 3
    {
        // Eğitim ve test dosya yollarını ve etiketlerini al
        var trainFilepaths = GetFilepaths(trainDataPath, out var trainLabels);
        var testFilepaths = GetFilepaths(testDataPath, out var testLabels);

        // Veriyi yükle ve ön işle
        var (trainImages, trainLabelsArray) = LoadData(trainFilepaths, trainLabels, targetWidth, targetHeight);
        var (testImages, testLabelsArray) = LoadData(testFilepaths, testLabels, targetWidth, targetHeight);

        // Modeli eğit
        var history = model.fit(trainImages, trainLabelsArray, epochs: epochs, shuffle: true);

        // Test sonuçlarını değerlendir
        var testResults = model.evaluate(testImages, testLabelsArray);

        // Test sonuçlarını ekrana yazdır
        if (testResults.ContainsKey("loss") && testResults.ContainsKey("accuracy"))
        {
            float testLoss = testResults["loss"];
            float testAccuracy = testResults["accuracy"];

            Console.WriteLine($"Test Loss: {testLoss}");
            Console.WriteLine($"Test Accuracy: {testAccuracy}");
        }
        else
        {
            Console.WriteLine("Test Sonuçları Sözlüğü beklenen anahtarları içermiyor.");
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Eğitim ve test veri yollarını belirt
        string trainDataPath = "C:/Users/accag/OneDrive/Masaüstü/train";
        string testDataPath = "C:/Users/accag/OneDrive/Masaüstü/test";

        // CardiomegalyPrediction sınıfından bir örnek oluştur
        var cardiomegalyPrediction = new CardiomegalyPrediction(trainDataPath, testDataPath);
        // Modeli oluştur
        cardiomegalyPrediction.BuildModel();
        // Modeli eğit ve değerlendir
        cardiomegalyPrediction.TrainAndEvaluate();
    }
}