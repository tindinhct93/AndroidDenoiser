package com.mobileml.shubham0204.scikitlearndemo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.R.attr.shape
import android.content.Context
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.nio.FloatBuffer
import java.util.*
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the views
        val inputEditTextView = findViewById<TextView>( R.id.textView )
        val outputTextView = findViewById<TextView>( R.id.output_textview )
        val secondTextView = findViewById<TextView>( R.id.time_textview )
        val predict_button = findViewById<Button>( R.id.predict_button )

        // Initialize the ORT environment and create the ORT session.
        val ortEnvironment = OrtEnvironment.getEnvironment()
        val ortSession = createORTSession( ortEnvironment )

        val context: Context = this // Obtain the current context from the activity
        val wavFile = File(context.resources.openRawResource(R.raw.debug).use { inputStream ->
            val file = File.createTempFile("temp", ".wav")
            file.outputStream().use { outputStream ->
                inputStream.copyTo(outputStream)
            }
            file.absolutePath
        })

        predict_button.setOnClickListener {
            // Parse input from inputEditText
            val FloatTensor = readWavFileToTensor(wavFile)

            val output = runFilePrediction( ortSession , ortEnvironment ,FloatTensor)
            val i= 12000
            outputTextView.text = "Inference Time is ${output[0]}us. Output is ${output[i+1]},${output[i+2]},${output[i+3]},${output[i+4]},${output[i+5]},${output[i+6]},${output[i+7]},${output[i+8]}. "
            secondTextView.text = output.size.toString()
            inputEditTextView.text = FloatTensor.size.toString()
        }
    }

    /**
     * This method reads a WAV file and converts it into a 1-second tensor of floats.
     * @param wavFile - the input WAV file
     * @return the tensor of floats
     */
    fun readWavFileToTensor(wavFile: File): FloatArray {
        val inputStream = FileInputStream(wavFile)

        val header = ByteArray(44)
        inputStream.read(header)

        val sampleRate = ByteBuffer.wrap(header, 24, 4).order(ByteOrder.LITTLE_ENDIAN).int
        val numChannels = ByteBuffer.wrap(header, 22, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt()

        val data = ByteArray(wavFile.length().toInt() - 44)
        inputStream.read(data)

        var max_length = 16000
        if (data.size/2< max_length) max_length = data.size/2
        val floatData = FloatArray(max_length)
        for (i in floatData.indices) {
            floatData[i] = ByteBuffer.wrap(data, i * 2, 2).order(ByteOrder.LITTLE_ENDIAN).short.toFloat() / Short.MAX_VALUE
        }
        return floatData
    }

    /**
     * This method reads a text file and returns a float array containing the values in the file.
     * @return the float array containing the values in the file
     */
    private fun readTXT() : FloatArray {
        val inputStream = resources.openRawResource(R.raw.tensor)

        val reader = BufferedReader(InputStreamReader(inputStream))

        val floatList = mutableListOf<Float>()

        try {
            // Read each line of the file and convert it to a float
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val value = line!!.toFloat()
                floatList.add(value)
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }

        val floatArray = FloatArray(floatList.size)
        for (i in floatList.indices) {
            floatArray[i] = floatList[i]
        }
        return floatArray
    }

    // Create an OrtSession
    private fun createORTSession( ortEnvironment: OrtEnvironment ) : OrtSession {
        val modelBytes = resources.openRawResource( R.raw.dynamic_model ).readBytes()
        return ortEnvironment.createSession( modelBytes )
    }

    //run prediction with the input from files.
    private fun runFilePrediction( ortSession: OrtSession , ortEnvironment: OrtEnvironment, InputFloatAraay: FloatArray ) : FloatArray {
        // Get the name of the input node
        val inputName = ortSession.inputNames?.iterator()?.next()

        val floatBuffer = FloatBuffer.wrap(InputFloatAraay)

        // Create input tensor with floatBufferInputs of shape ( 1 , 1 )
        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment, floatBuffer, longArrayOf(
                1, 1, InputFloatAraay.size.toLong()
            )
        )
        // Run the model

        val tStart = System.currentTimeMillis()
        val results = ortSession.run(mapOf(inputName to inputTensor))

        val tEnd = System.currentTimeMillis()
        val tDelta = tEnd - tStart

        val ouput = results[0].value as Array<Array<FloatArray>>
        val float_waveform = ouput[0][0]
        float_waveform[0] = tDelta.toFloat()

        return float_waveform
    }

    // Make predictions with dummy inputs
    private fun runPrediction( ortSession: OrtSession , ortEnvironment: OrtEnvironment, rand: Random ) : FloatArray {
        // Get the name of the input node
        val inputName = ortSession.inputNames?.iterator()?.next()
        // Make a FloatBuffer of the inputs
        var numbers = FloatArray(44100)

        for (i in 0..44099) {
            numbers[i] = 1.0f
        }
        numbers = readTXT()

        val floatBuffer = FloatBuffer.wrap(numbers)

        // Create input tensor with floatBufferInputs of shape ( 1 , 1 )
        val inputTensor = OnnxTensor.createTensor(
            ortEnvironment, floatBuffer, longArrayOf(
                1, 1, 44100
            )
        )

        // Run the model
        val tStart = System.currentTimeMillis()
        val results = ortSession.run(mapOf(inputName to inputTensor))

        for (i in 1..9) {
            ortSession.run(mapOf(inputName to inputTensor))
        }
        // Fetch and return the results
        val tEnd = System.currentTimeMillis()
        val tDelta = tEnd - tStart

        val ouput = results[0].value as Array<Array<FloatArray>>
        val float_waveform = ouput[0][0]
        float_waveform[0] = tDelta.toFloat()

        return float_waveform
    }
}
