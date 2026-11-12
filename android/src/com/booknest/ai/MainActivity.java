package com.booknest.ai;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebSettings;
import android.webkit.WebViewClient;
import android.webkit.WebChromeClient;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.JavascriptInterface;
import android.accounts.Account;
import android.accounts.AccountManager;
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
    private final static int GOOGLE_ACCOUNT_PICKER_REQUEST = 1001;
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

        // Native Android Bridge for Real Phone Google Accounts
        mWebView.addJavascriptInterface(new AndroidAuthBridge(this), "AndroidGoogleAuth");

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
        if (requestCode == GOOGLE_ACCOUNT_PICKER_REQUEST) {
            android.util.Log.d("BookNestAuth", "Picker resultCode=" + resultCode + " data=" + data);
            if (resultCode == Activity.RESULT_OK && data != null) {
                String accountName = data.getStringExtra(AccountManager.KEY_ACCOUNT_NAME);
                if (accountName == null || accountName.trim().isEmpty()) {
                    accountName = data.getStringExtra("authAccount");
                }
                if (accountName == null || accountName.trim().isEmpty()) {
                    accountName = data.getStringExtra("accountName");
                }
                if ((accountName == null || accountName.trim().isEmpty()) && data.getExtras() != null) {
                    for (String key : data.getExtras().keySet()) {
                        Object val = data.getExtras().get(key);
                        android.util.Log.d("BookNestAuth", "Intent extra: " + key + " = " + val);
                        if (val instanceof String && ((String) val).contains("@")) {
                            accountName = (String) val;
                            break;
                        }
                    }
                }
                android.util.Log.d("BookNestAuth", "Resolved accountName=" + accountName);
                if (accountName != null && !accountName.trim().isEmpty()) {
                    final String chosenEmail = accountName.trim();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            String js = "(function() {" +
                                "if (typeof window.onNativeGoogleAccountSelected === 'function') {" +
                                "    window.onNativeGoogleAccountSelected('" + chosenEmail.replace("'", "\\'") + "');" +
                                "} else if (typeof performGoogleLogin === 'function') {" +
                                "    performGoogleLogin('', '" + chosenEmail.replace("'", "\\'") + "', '');" +
                                "}" +
                                "})();";
                            mWebView.evaluateJavascript(js, null);
                        }
                    });
                }
            } else {
                android.util.Log.d("BookNestAuth", "Account picker cancelled or no account picked.");
            }
        } else if (requestCode == FILECHOOSER_RESULTCODE) {
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

    public class AndroidAuthBridge {
        private final MainActivity activity;

        public AndroidAuthBridge(MainActivity activity) {
            this.activity = activity;
        }

        @JavascriptInterface
        public String getPhoneGoogleAccountsJson() {
            try {
                Account[] accounts = AccountManager.get(activity).getAccountsByType("com.google");
                if (accounts != null && accounts.length > 0) {
                    StringBuilder sb = new StringBuilder("[");
                    for (int i = 0; i < accounts.length; i++) {
                        if (accounts[i] != null && accounts[i].name != null) {
                            if (i > 0) sb.append(",");
                            sb.append("\"").append(accounts[i].name.replace("\"", "\\\"")).append("\"");
                        }
                    }
                    sb.append("]");
                    android.util.Log.d("BookNestAuth", "Found phone Google accounts: " + sb.toString());
                    return sb.toString();
                }
            } catch (Exception e) {
                android.util.Log.e("BookNestAuth", "Failed querying phone Google accounts", e);
            }
            return "[]";
        }

        @JavascriptInterface
        public void openGoogleAccountChooser() {
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        Account selectedAccount = null;
                        try {
                            Account[] accounts = AccountManager.get(activity).getAccountsByType("com.google");
                            if (accounts != null && accounts.length > 0) {
                                selectedAccount = accounts[0];
                            }
                        } catch (Exception ignored) {}

                        Intent intent = AccountManager.newChooseAccountIntent(
                            selectedAccount,
                            null,
                            new String[]{"com.google"},
                            "Select your Google Account for BookNest AI",
                            null,
                            null,
                            null
                        );
                        activity.startActivityForResult(intent, GOOGLE_ACCOUNT_PICKER_REQUEST);
                    } catch (Exception e) {
                        android.util.Log.e("BookNestWeb", "Failed to launch Google account picker", e);
                    }
                }
            });
        }

        @JavascriptInterface
        public void exitApp() {
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    activity.finish();
                }
            });
        }

        @JavascriptInterface
        public boolean isNativeApp() {
            return true;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mWebView != null) {
            mWebView.evaluateJavascript("if (typeof window.redirectToHomePage === 'function') { window.redirectToHomePage(); }", null);
        }
    }

    @Override
    public void onBackPressed() {
        if (mWebView != null) {
            mWebView.evaluateJavascript(
                "(function() {" +
                "  if (typeof window.handleAppBackPress === 'function') {" +
                "    return window.handleAppBackPress();" +
                "  }" +
                "  return 'unhandled';" +
                "})();",
                new ValueCallback<String>() {
                    @Override
                    public void onReceiveValue(String value) {
                        if ("\"unhandled\"".equals(value) || value == null || "null".equals(value)) {
                            showNativeExitDialog();
                        }
                    }
                }
            );
        } else {
            showNativeExitDialog();
        }
    }

    private void showNativeExitDialog() {
        new android.app.AlertDialog.Builder(this)
            .setTitle("Exit BookNest AI")
            .setMessage("Are you sure you want to exit? Your books and reading progress are safely preserved.")
            .setPositiveButton("Exit", new android.content.DialogInterface.OnClickListener() {
                @Override
                public void onClick(android.content.DialogInterface dialog, int which) {
                    finish();
                }
            })
            .setNegativeButton("Keep Reading", new android.content.DialogInterface.OnClickListener() {
                @Override
                public void onClick(android.content.DialogInterface dialog, int which) {
                    dialog.dismiss();
                }
            })
            .setCancelable(true)
            .show();
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
