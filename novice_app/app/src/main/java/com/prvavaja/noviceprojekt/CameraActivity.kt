package com.prvavaja.noviceprojekt

import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.prvavaja.noviceprojekt.databinding.ActivityCameraActivtyBinding
import io.realm.kotlin.ext.realmListOf
import io.realm.kotlin.types.RealmInstant
import org.eclipse.paho.client.mqttv3.MqttClient
import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import org.eclipse.paho.client.mqttv3.MqttMessage
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class CameraActivity : AppCompatActivity() {
    private lateinit var binding: ActivityCameraActivtyBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private lateinit var myApplication: MyAplication

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraActivtyBinding.inflate(layoutInflater)
        setContentView(binding.root)
        myApplication = application as MyAplication

        cameraExecutor = Executors.newSingleThreadExecutor()
        binding.btnDisplayBackCamera.setOnClickListener(){
            finish()
        }

        binding.takePhotoButton.setOnClickListener {
            val imageCapture = imageCapture ?: return@setOnClickListener

            imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        val buffer = image.planes[0].buffer
                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)

                        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                        val byteArrayOutputStream = ByteArrayOutputStream()
                        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
                        val byteArray = byteArrayOutputStream.toByteArray()
                        val encodedImage: String = Base64.encodeToString(byteArray, Base64.DEFAULT)

                        val qos = 2
                        val clientId = UUID.randomUUID().toString()
                        val client = MqttClient("tcp://10.0.2.2:1883", clientId, MemoryPersistence())
                        val connOpts = MqttConnectOptions()
                        connOpts.isCleanSession = true
                        client.connect(connOpts)
                        val message = MqttMessage(encodedImage.toByteArray())
                        message.qos = qos
                        client.publish("image/find-text-input", message)

                        client.subscribe(
                            "image/find-text-response", qos
                        ) { topic, messageResponse ->
                            writeImageToServer(String(messageResponse.payload))
                        }

                        image.close()
                    }

                    override fun onError(exception: ImageCaptureException) {
                        super.onError(exception)
                        Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                    }
                }
            )
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
    }
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            imageCapture = ImageCapture.Builder().build()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture)
            } catch(e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                finish()
            }
        }
    }


    companion object {
        private const val TAG = "CameraXApp"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA, android.Manifest.permission.INTERNET)
    }

    private fun writeImageToServer(image: String) {
        println("Writing result image to server")
        try {
            myApplication.realm.writeBlocking {
                var article = NewsArticleRealm()
                article.authors = realmListOf("senzor")
                article.date = RealmInstant.now()
                article.categories = realmListOf("senzor-prepoznava-teksta")
                article.content = image
                article.title = "Prepoznava teksta iz senzorja"
                article.location = LocationRealm()
                article.location!!.type = GenerateDataActivity.realLastLocation.type
                println(GenerateDataActivity.realLastLocation.coordinates)
                article.location!!.coordinates = realmListOf(
                    GenerateDataActivity.realLastLocation.coordinates.first,
                    GenerateDataActivity.realLastLocation.coordinates.second
                )
                copyToRealm(article)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
