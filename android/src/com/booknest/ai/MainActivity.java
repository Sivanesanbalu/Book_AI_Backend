package com.booknest.ai;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebSettings;
import android.webkit.WebViewClient;
import android.webkit.WebChromeClient;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.net.Uri;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.Manifest;
import android.graphics.Color;
import android.view.Window;
import android.view.WindowManager;

public class MainActivity extends Activity {

    private WebView mWebView;
    public ValueCallback<Uri[]> mUploadMessage;
    public final static int FILECHOOSER_RESULTCODE = 1;
    private final static int CAMERA_PERMISSION_CODE = 100;
    private static final String APP_URL = "https://relationship-cruises-garcia-kelkoo.trycloudflare.com";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        mWebView = new WebView(this);
        mWebView.setBackgroundColor(Color.parseColor("#0a0d14"));
        setContentView(mWebView);

        WebView.setWebContentsDebuggingEnabled(true);

        WebSettings settings = mWebView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setDatabaseEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setAllowFileAccess(true);
        settings.setAllowContentAccess(true);
        settings.setUseWideViewPort(true);
        settings.setLoadWithOverviewMode(true);
        settings.setJavaScriptCanOpenWindowsAutomatically(true);
        settings.setCacheMode(WebSettings.LOAD_DEFAULT);

        mWebView.setWebViewClient(new AppWebViewClient());
        mWebView.setWebChromeClient(new AppWebChromeClient(this));

        // Always start loading main page immediately
        mWebView.loadUrl(APP_URL);

        // Request camera permission if not granted
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == FILECHOOSER_RESULTCODE) {
            if (mUploadMessage == null) return;
            Uri[] results = null;
            if (resultCode == Activity.RESULT_OK && data != null) {
                String dataString = data.getDataString();
                if (dataString != null) {
                    results = new Uri[]{Uri.parse(dataString)};
                }
            }
            mUploadMessage.onReceiveValue(results);
            mUploadMessage = null;
        } else {
            super.onActivityResult(requestCode, resultCode, data);
        }
    }

    @Override
    public void onBackPressed() {
        if (mWebView != null && mWebView.canGoBack()) {
            mWebView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    static class AppWebViewClient extends WebViewClient {
        @Override
        public boolean shouldOverrideUrlLoading(WebView view, String url) {
            view.loadUrl(url);
            return true;
        }

        @Override
        public void onReceivedError(WebView view, int errorCode, String description, String failingUrl) {
            if (failingUrl != null && (failingUrl.startsWith(APP_URL) || failingUrl.equals(APP_URL) || failingUrl.equals(APP_URL + "/"))) {
                String errorHtml = "<!DOCTYPE html><html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'>" +
                    "<style>body{background:#0a0d14;color:#f8fafc;font-family:system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;margin:0;padding:24px;box-sizing:border-box;text-align:center;}" +
                    ".card{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);padding:24px;border-radius:20px;max-width:320px;box-shadow:0 12px 36px rgba(0,0,0,0.4);}" +
                    "h2{margin:0 0 10px 0;font-size:1.2rem;color:#f43f5e;font-weight:800;}" +
                    "p{font-size:0.85rem;color:#94a3b8;line-height:1.5;margin-bottom:20px;}" +
                    "button{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;border:none;padding:12px 24px;border-radius:12px;font-weight:700;cursor:pointer;font-size:0.9rem;box-shadow:0 4px 16px rgba(99,102,241,0.4);}" +
                    "</style></head><body>" +
                    "<div class='card'>" +
                    "<h2>Connection Notice</h2>" +
                    "<p>Unable to connect to BookNest AI server. Please make sure the host is running and tap below to reconnect.</p>" +
                    "<button onclick=\"window.location.href='" + APP_URL + "'\">🔄 Tap to Reconnect</button>" +
                    "</div></body></html>";
                view.loadDataWithBaseURL(null, errorHtml, "text/html", "UTF-8", null);
            }
        }
    }

    static class AppWebChromeClient extends WebChromeClient {
        private final MainActivity activity;

        public AppWebChromeClient(MainActivity activity) {
            this.activity = activity;
        }

        @Override
        public boolean onConsoleMessage(android.webkit.ConsoleMessage consoleMessage) {
            android.util.Log.d("BookNestWeb", consoleMessage.message() + " [" + consoleMessage.sourceId() + ":" + consoleMessage.lineNumber() + "]");
            return true;
        }

        @Override
        public void onPermissionRequest(final PermissionRequest request) {
            activity.runOnUiThread(new PermissionGranter(request));
        }

        @Override
        public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> filePathCallback, FileChooserParams fileChooserParams) {
            if (activity.mUploadMessage != null) {
                activity.mUploadMessage.onReceiveValue(null);
            }
            activity.mUploadMessage = filePathCallback;

            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("image/*");
            activity.startActivityForResult(Intent.createChooser(intent, "Select Book Cover"), FILECHOOSER_RESULTCODE);
            return true;
        }
    }

    static class PermissionGranter implements Runnable {
        private final PermissionRequest request;

        public PermissionGranter(PermissionRequest request) {
            this.request = request;
        }

        @Override
        public void run() {
            request.grant(request.getResources());
        }
    }
}
