package com.example.novinar

import android.app.Dialog
import android.graphics.BitmapFactory
import android.os.Bundle
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

        arguments?.getParcelable<News>(NEWS_KEY)?.let { news ->
            titleTextView.text = news.title
            contentTextView.text = news.content ?: "No content available"
            categoryTextView.text = news.categories?.joinToString(", ") ?: "No category"
            timestampTextView.text = news.timestamp ?: "No timestamp"

            news.location?.coordinates?.let {
                Log.d("DetailViewFragment", "News Location: Lat = ${it[1]}, Long = ${it[0]}")
            }

            news.url?.let { imageUrl ->
                try {

                } catch (e: Exception) {
                    Log.e("DetailViewFragment", "Error loading image: ${e.localizedMessage}")
                    imageView.setImageResource(R.drawable.placeholder_image)
                }
            } ?: run {
                imageView.setImageResource(R.drawable.placeholder_image)
            }
        }

        return view
    }
}
