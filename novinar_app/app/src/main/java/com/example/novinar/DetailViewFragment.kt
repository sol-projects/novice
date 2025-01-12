package com.example.novinar

import android.os.Bundle
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

        arguments?.getParcelable<News>(NEWS_KEY)?.let { news ->
            titleTextView.text = news.title
            contentTextView.text = news.content
            categoryTextView.text = news.category
            timestampTextView.text = news.timestamp
            // Set image if available
            news.image?.let { imageBase64 ->
                val imageBytes = android.util.Base64.decode(imageBase64.split(",")[1], android.util.Base64.DEFAULT)
                val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                imageView.setImageBitmap(bitmap)
            }
        }

        return view
    }
}
