// Generated by view binder compiler. Do not edit!
package com.prvavaja.noviceprojekt.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.Toolbar;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.prvavaja.noviceprojekt.R;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class ActivityMainBinding implements ViewBinding {
  @NonNull
  private final ConstraintLayout rootView;

  @NonNull
  public final ImageButton btnSetings;

  @NonNull
  public final Button buttonCamera;

  @NonNull
  public final RecyclerView recyclerView;

  @NonNull
  public final Toolbar toolbar;

  @NonNull
  public final TextView tvTitle;

  private ActivityMainBinding(@NonNull ConstraintLayout rootView, @NonNull ImageButton btnSetings,
      @NonNull Button buttonCamera, @NonNull RecyclerView recyclerView, @NonNull Toolbar toolbar,
      @NonNull TextView tvTitle) {
    this.rootView = rootView;
    this.btnSetings = btnSetings;
    this.buttonCamera = buttonCamera;
    this.recyclerView = recyclerView;
    this.toolbar = toolbar;
    this.tvTitle = tvTitle;
  }

  @Override
  @NonNull
  public ConstraintLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static ActivityMainBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static ActivityMainBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.activity_main, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static ActivityMainBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.btnSetings;
      ImageButton btnSetings = ViewBindings.findChildViewById(rootView, id);
      if (btnSetings == null) {
        break missingId;
      }

      id = R.id.buttonCamera;
      Button buttonCamera = ViewBindings.findChildViewById(rootView, id);
      if (buttonCamera == null) {
        break missingId;
      }

      id = R.id.recyclerView;
      RecyclerView recyclerView = ViewBindings.findChildViewById(rootView, id);
      if (recyclerView == null) {
        break missingId;
      }

      id = R.id.toolbar;
      Toolbar toolbar = ViewBindings.findChildViewById(rootView, id);
      if (toolbar == null) {
        break missingId;
      }

      id = R.id.tvTitle;
      TextView tvTitle = ViewBindings.findChildViewById(rootView, id);
      if (tvTitle == null) {
        break missingId;
      }

      return new ActivityMainBinding((ConstraintLayout) rootView, btnSetings, buttonCamera,
          recyclerView, toolbar, tvTitle);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}