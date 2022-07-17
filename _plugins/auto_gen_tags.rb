# This plugin generates tag pages automatically after saving posts.
#
# To enable this plugin, please do the following steps:
# 1, Create a dir named 'tags' under the root dir.
# 2. Create a layout, named `_layouts/tag.html,` for tag pages. For example,
#   ---
#   layout: default
#   ---
#   <h1>
#     Tag: {{ page.tag }}
#   </h1>
#   <ul>
#   	{% for post in site.posts %}
#   		{% if post.tags contains page.tag %}
#   			<li>
#   			    <a href="{{ post.url }}">{{ post.title }}</a> 
#   			    ({{ post.date | date_to_string }})<br>
#   		    </li>
#   		{% endif %}
#   	{% endfor %}
#   </ul>
#
# 3. Comment out the `return` statement in the next line.
return
# 4. Start jekyll server at local, and the server will parse all posts and
#    generate tag pages to the `tags/` dir. The tag pages are also generated
#    on-the-fly after saving posts if jekyll server is running at background.

Jekyll::Hooks.register :posts, :post_write do |post|
  all_existing_tags = Dir.entries("tags")
    .map { |t| t.match(/(.*).md/) }
    .compact.map { |m| m[1] }

  tags = post['tags'].reject { |t| t.empty? }
  tags.each do |tag|
    generate_tag_file(tag) if !all_existing_tags.include?(tag)
  end
end

def generate_tag_file(tag)
  File.open("tags/#{tag}.md", "wb") do |file|
    file << "---\nlayout: tag\ntitle: \"Tag: #{tag}\"\ntag: #{tag}\n---\n"
  end
end

