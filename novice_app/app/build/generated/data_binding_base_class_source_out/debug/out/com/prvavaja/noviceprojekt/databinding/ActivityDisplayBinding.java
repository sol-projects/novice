// Generated by view binder compiler. Do not edit!
package com.prvavaja.noviceprojekt.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.Toolbar;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.prvavaja.noviceprojekt.R;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class ActivityDisplayBinding implements ViewBinding {
  @NonNull
  private final ConstraintLayout rootView;

  @NonNull
  public final ImageButton btnDisplayBack;

  @NonNull
  public final TextView contentTextView;

  @NonNull
  public final TextView mainTitle;

  @NonNull
  public final TextView newsTitle;

  @NonNull
  public final TextView textViewDate;

  @NonNull
  public final TextView textViewDisplayAuthor;

  @NonNull
  public final TextView textViewDisplayCatagories;

  @NonNull
  public final TextView textViewDisplayLocation;

  @NonNull
  public final TextView textViewURL;

  @NonNull
  public final Toolbar toolbar;

  private ActivityDisplayBinding(@NonNull ConstraintLayout rootView,
      @NonNull ImageButton btnDisplayBack, @NonNull TextView contentTextView,
      @NonNull TextView mainTitle, @NonNull TextView newsTitle, @NonNull TextView textViewDate,
      @NonNull TextView textViewDisplayAuthor, @NonNull TextView textViewDisplayCatagories,
      @NonNull TextView textViewDisplayLocation, @NonNull TextView textViewURL,
      @NonNull Toolbar toolbar) {
    this.rootView = rootView;
    this.btnDisplayBack = btnDisplayBack;
    this.contentTextView = contentTextView;
    this.mainTitle = mainTitle;
    this.newsTitle = newsTitle;
    this.textViewDate = textViewDate;
    this.textViewDisplayAuthor = textViewDisplayAuthor;
    this.textViewDisplayCatagories = textViewDisplayCatagories;
    this.textViewDisplayLocation = textViewDisplayLocation;
    this.textViewURL = textViewURL;
    this.toolbar = toolbar;
  }

  @Override
  @NonNull
  public ConstraintLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static ActivityDisplayBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static ActivityDisplayBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.activity_display, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static ActivityDisplayBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.btnDisplayBack;
      ImageButton btnDisplayBack = ViewBindings.findChildViewById(rootView, id);
      if (btnDisplayBack == null) {
        break missingId;
      }

      id = R.id.contentTextView;
      TextView contentTextView = ViewBindings.findChildViewById(rootView, id);
      if (contentTextView == null) {
        break missingId;
      }

      id = R.id.mainTitle;
      TextView mainTitle = ViewBindings.findChildViewById(rootView, id);
      if (mainTitle == null) {
        break missingId;
      }

      id = R.id.newsTitle;
      TextView newsTitle = ViewBindings.findChildViewById(rootView, id);
      if (newsTitle == null) {
        break missingId;
      }

      id = R.id.textViewDate;
      TextView textViewDate = ViewBindings.findChildViewById(rootView, id);
      if (textViewDate == null) {
        break missingId;
      }

      id = R.id.textViewDisplayAuthor;
      TextView textViewDisplayAuthor = ViewBindings.findChildViewById(rootView, id);
      if (textViewDisplayAuthor == null) {
        break missingId;
      }

      id = R.id.textViewDisplayCatagories;
      TextView textViewDisplayCatagories = ViewBindings.findChildViewById(rootView, id);
      if (textViewDisplayCatagories == null) {
        break missingId;
      }

      id = R.id.textViewDisplayLocation;
      TextView textViewDisplayLocation = ViewBindings.findChildViewById(rootView, id);
      if (textViewDisplayLocation == null) {
        break missingId;
      }

      id = R.id.textViewURL;
      TextView textViewURL = ViewBindings.findChildViewById(rootView, id);
      if (textViewURL == null) {
        break missingId;
      }

      id = R.id.toolbar;
      Toolbar toolbar = ViewBindings.findChildViewById(rootView, id);
      if (toolbar == null) {
        break missingId;
      }

      return new ActivityDisplayBinding((ConstraintLayout) rootView, btnDisplayBack,
          contentTextView, mainTitle, newsTitle, textViewDate, textViewDisplayAuthor,
          textViewDisplayCatagories, textViewDisplayLocation, textViewURL, toolbar);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}