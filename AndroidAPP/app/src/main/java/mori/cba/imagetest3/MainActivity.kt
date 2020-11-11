package mori.cba.imagetest3

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment.DIRECTORY_PICTURES
import android.provider.MediaStore
import android.util.Base64
import android.util.Log
import android.view.View
import android.view.Window
import android.view.WindowManager
import android.webkit.MimeTypeMap
import android.widget.ImageView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.android.synthetic.main.activity_main.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.*



private const val FILE_NAME = "photo.jpg"
private const val REQUEST_CODE = 42
private const val GALLERY_REQUEST_CODE = 105
var selectedImage: ImageView? = null
private lateinit var photoFile: File
var imageString = ""
var encodedString = ""
var prediction = ""

class MainActivity : AppCompatActivity() {

    @RequiresApi(Build.VERSION_CODES.LOLLIPOP)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        getSupportActionBar()?.hide()
        selectedImage = findViewById<ImageView>(R.id.imageView)
        val window: Window = window
        window.addFlags(WindowManager.LayoutParams.FLAG_DRAWS_SYSTEM_BAR_BACKGROUNDS)
        val colorCodeDark: Int = Color.parseColor("#44A9FA")
        window.setStatusBarColor(colorCodeDark)


        btnTakePicture.setOnClickListener {
            val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            photoFile = getphotoFile(FILE_NAME)
            //takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoFile) this doesnt work for api>23
            val fileProvider = FileProvider.getUriForFile(this,"mori.cba.imagetest3.fileprovider", photoFile)
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)
            if (takePictureIntent.resolveActivity(this.packageManager)!=null){
                startActivityForResult(takePictureIntent, REQUEST_CODE)
            }else {
                Toast.makeText(this, "Unable to open camera", Toast.LENGTH_SHORT).show()
            }


        }
        btnChoosePicture.setOnClickListener(View.OnClickListener {
            val gallery =
                Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(gallery, GALLERY_REQUEST_CODE)
        })
        PredictButtom.setOnClickListener(){
            initPython()
            val bm = (imageView.drawable as BitmapDrawable).bitmap
            imageString = getStringFromImage(bm)
            encodedString = getLungsHeatmap(imageString)
            val imageBitmap = getBitmapFromString(encodedString )
            imageView.setImageBitmap(imageBitmap)
            prediction = getDiagnose(imageString)
            textView.text = prediction
        }

    }

    private fun getBitmapFromString(encodedString: String): Bitmap {
        val data = android.util.Base64.decode(encodedString,Base64.DEFAULT)
        val bmp = BitmapFactory.decodeByteArray(data,0,data.size)
        return bmp
    }

    private fun getLungsHeatmap(imageString: String): String {
        val python = Python.getInstance()
        val pythonFile = python.getModule("lungs")
        return  pythonFile.callAttr("main",imageString).toString()
    }

    private fun getDiagnose(imageString: String): String {
        val python = Python.getInstance()
        val pythonFile = python.getModule("lungs")
        return  pythonFile.callAttr("pred",imageString).toString()
    }

    private fun getStringFromImage(bm: Bitmap?): String {
        var baos = ByteArrayOutputStream()
        bm?.compress(Bitmap.CompressFormat.PNG,100, baos )
        var imageBytes = baos.toByteArray()
        var Image_encoded = Base64.encodeToString(imageBytes,Base64.DEFAULT)
        return Image_encoded


    }

    private fun getphotoFile(fileName: String): File {
        val storageDirectory = getExternalFilesDir(DIRECTORY_PICTURES)
        return File.createTempFile(fileName,".jpg",storageDirectory)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_CODE && resultCode == Activity.RESULT_OK){
            //val takenImage = data?.extras?.get("data") as Bitmap
            val takenImage = BitmapFactory.decodeFile(photoFile.absolutePath)
            imageView.setImageBitmap(takenImage)
        } else {
            super.onActivityResult(requestCode, resultCode, data)
        }

        if (requestCode == GALLERY_REQUEST_CODE) {
            if (resultCode == Activity.RESULT_OK) {
                val contentUri = data!!.data
                val timeStamp =
                    SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
                val takenImage =
                    "JPEG_" + timeStamp + "." + contentUri?.let { getFileExt(it) }
                Log.d("tag", "onActivityResult: Gallery Image Uri:  $takenImage")
                selectedImage?.setImageURI(contentUri)
            }
        }
    }

    private fun getFileExt(contentUri: Uri): String? {
        val c = contentResolver
        val mime = MimeTypeMap.getSingleton()
        return mime.getExtensionFromMimeType(c.getType(contentUri))
    }

    private fun initPython(){
        if (! Python.isStarted()) {
            Python.start(AndroidPlatform(this));
        }
    }



}