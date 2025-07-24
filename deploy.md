1. Flask アプリを GitHub に push

2. Azure App Service を作成
Azure Portal で以下を設定：
- ランタイム：Python 3.x
- OS：Linux
- App Service プラン：F1

3. スタートアップの登録
- Azure Portal → App Service → 設定 → 構成 → スタートアップ コマンド　scripts/startup.sh

4. ビルド後処理の登録 
- Azure Portal → App Service → 設定 → 環境変数 → POST_BUILD_COMMAND=python -m pip install -r requirements.txt

5. Azure の発行プロファイルを GitHub に登録
- Azure Portal → App Service → 設定 → 構成 → SCM 基本認証の発行資格情報 ON
- Azure Portal → App Service → 「発行プロファイルを取得」
→ GitHub のリポジトリに AZURE_WEBAPP_PUBLISH_PROFILE という名前で Secrets に登録

6. GitHub Actions ワークフローを実行
.github/workflows/build_deploy.yml

7. デプロイ確認
- GitHub Actions のログで成功を確認し、
https://utilitytool-wadayym.azurewebsites.net にアクセスして動作確認

