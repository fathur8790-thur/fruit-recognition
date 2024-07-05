

package app.ij.mlwithtensorflowlite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import app.ij.mlwithtensorflowlite.ml.Model;

public class MainActivity extends AppCompatActivity {

    TextView result, benefits, sideEffects;
    ImageView imageView;
    Button picture;
    int imageSize = 224;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        benefits = findViewById(R.id.benefits); // Added for showing benefits
        sideEffects = findViewById(R.id.sideEffects); // Added for showing side effects
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 1);
                } else {
                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
    }

    public void classifyImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // get 1D array of 224 * 224 pixels in image
            int [] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            // find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i = 0; i < confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"APEL", "NANAS", "PEPAYA", "JAMBU", "JERUK"};
            String[] benefitsArray = {
                    "Menjaga kesehatan pencernaan, menurunkan risiko penyakit jantung, meningkatkan kesehatan otak, dan mengurangi risiko kanker.",
                    "Meningkatkan daya tahan tubuh, menjaga kesehatan tulang, membantu penyembuhan luka pasca operasi, dan menurunkan asam urat.",
                    "Melancarkan pencernaan, meningkatkan kesehatan kulit, mengurangi peradangan, dan meningkatkan fungsi ginjal.",
                    "Mengatur tekanan darah, menyehatkan kulit, membantu mengendalikan gula darah, dan menyegarkan napas.",
                    "Meningkatkan sistem imun, menurunkan kolesterol, mengurangi risiko batu ginjal, dan mendukung kesehatan kulit."
            };
            String[] sideEffectsArray = {
                    "Menyebabkan berat badan sulit turun, merusak gigi, dan membuat gula darah naik turun.",
                    "Membuat perut kembung, memicu gejala maag, dan menyebabkan lidah gatal dan panas.",
                    "Dapat merusak kerongkongan, menimbulkan efek pencahar seperti sakit perut dan diare, serta dapat memicu perut kembung.",
                    "Dapat mengganggu kinerja obat diabetes pada penderita diabetes dan potensi menyebabkan radang usus buntu.",
                    "Memperburuk gejala penyakit GERD serta dapat menyebabkan masalah pencernaan dan memicu reaksi alergi pada beberapa orang."
            };

            result.setText(classes[maxPos]);
            benefits.setText(benefitsArray[maxPos]); // Set the benefits text
            sideEffects.setText(sideEffectsArray[maxPos]); // Set the side effects text

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            int dimension = Math.min(image.getWidth(), image.getHeight());
            image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
            imageView.setImageBitmap(image);

            image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
            classifyImage(image);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}
