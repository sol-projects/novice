package com.example.novinar

import android.app.Dialog
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Base64
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.Fragment

class DetailViewFragment : Fragment() {

    companion object {
        private const val NEWS_KEY = "news"

        fun newInstance(news: News): DetailViewFragment {
            val fragment = DetailViewFragment()
            val args = Bundle()
            args.putParcelable(NEWS_KEY, news)
            fragment.arguments = args
            return fragment
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_detail_view, container, false)

        val titleTextView: TextView = view.findViewById(R.id.textViewTitle)
        val contentTextView: TextView = view.findViewById(R.id.textViewContent)
        val categoryTextView: TextView = view.findViewById(R.id.textViewCategory)
        val timestampTextView: TextView = view.findViewById(R.id.textViewTimestamp)
        val imageView: ImageView = view.findViewById(R.id.imageView)

        // Retrieve the news object from arguments
        arguments?.getParcelable<News>(NEWS_KEY)?.let { news ->
            // Populate the UI with news details
            titleTextView.text = news.title
            contentTextView.text = news.content ?: "No content available"
            categoryTextView.text = news.category ?: "No category"
            timestampTextView.text = news.timestamp ?: "No timestamp"

            // Decode and display the image if available
            news.image?.let { imageBase64 ->
                try {
                    val base64Image = if (imageBase64.contains(",")) {
                        imageBase64.split(",")[1]
                    } else {
                        imageBase64
                    }
                    val imageBytes = Base64.decode(base64Image, Base64.DEFAULT)
                    val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    imageView.setImageBitmap(bitmap)

                    // Add click listener to open the image in full screen
                    imageView.setOnClickListener {
                        showFullScreenImage(bitmap)
                    }
                } catch (e: Exception) {
                    Log.e("DetailViewFragment", "Error decoding image: ${e.localizedMessage}")
                    imageView.setImageResource(R.drawable.placeholder_image) // Fallback image
                }
            } ?: run {
                imageView.setImageResource(R.drawable.placeholder_image) // Fallback image
            }
        }

        return view
    }

    private fun showFullScreenImage(bitmap: android.graphics.Bitmap) {
        val dialog = Dialog(requireContext(), android.R.style.Theme_Black_NoTitleBar_Fullscreen)
        dialog.setContentView(R.layout.dialog_fullscreen_image)
        val fullScreenImageView: ImageView = dialog.findViewById(R.id.fullScreenImageView)
        fullScreenImageView.setImageBitmap(bitmap)
        fullScreenImageView.setOnClickListener {
            dialog.dismiss() // Close the dialog when the image is clicked
        }
        dialog.show()
    }
}
