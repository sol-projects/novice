package com.example.novinar

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.Fragment
import java.io.File

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

    private fun formatTimestamp(timestamp: String?): String {
        return try {
            val timeMillis = timestamp?.toLongOrNull() ?: return "Invalid timestamp"
            val dateFormat =
                java.text.SimpleDateFormat("dd MMM yyyy, HH:mm", java.util.Locale.getDefault())
            val date = java.util.Date(timeMillis)
            dateFormat.format(date)
        } catch (e: Exception) {
            Log.e("DetailViewFragment", "Error formatting timestamp: ${e.localizedMessage}")
            "Invalid timestamp"
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

        arguments?.getParcelable<News>(NEWS_KEY)?.let { news ->
            titleTextView.text = news.title
            contentTextView.text = news.content

            categoryTextView.text = news.categories?.joinToString(", ") ?: "No category provided"

            val timestamp = news.authors?.firstOrNull()
            timestampTextView.text = formatTimestamp(timestamp)

            news.url?.let { imagePath ->
                val imageFile = File(imagePath)
                if (imageFile.exists()) {
                    val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
                    imageView.setImageBitmap(bitmap)
                } else {
                    Log.e("DetailViewFragment", "Image file not found: $imagePath")
                    imageView.setImageResource(R.drawable.placeholder_image)
                }
            } ?: run {
                imageView.setImageResource(R.drawable.placeholder_image)
            }
        }

        return view
    }

}
