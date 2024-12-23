package com.example.novinar

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class ViewNewsFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_view_news, container, false)
        val recyclerView: RecyclerView = view.findViewById(R.id.recyclerViewNews)

        NewsRepository.loadNewsList()

        val newsList = NewsRepository.getNewsList()

        val adapter = NewsAdapter(
            newsList = newsList,
            onEdit = { index ->
                val news = newsList[index]
                val bundle = Bundle().apply {
                    putInt("news_index", index)
                    putString("news_title", news.title)
                    putString("news_content", news.content)
                }
                val fragment = PostNewsFragment().apply { arguments = bundle }
                requireActivity().supportFragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, fragment)
                    .addToBackStack(null)
                    .commit()
            },
            onDelete = { index ->
                NewsRepository.deleteNews(index)
                recyclerView.adapter?.notifyItemRemoved(index)
            }
        )


        recyclerView.layoutManager = LinearLayoutManager(requireContext())
        recyclerView.adapter = adapter

        return view
    }
}
